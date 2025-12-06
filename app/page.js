"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [status, setStatus] = useState("Idle");
  const [isStarting, setIsStarting] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [motionLevel, setMotionLevel] = useState(0); // 0 to ~1
  const [livenessScore, setLivenessScore] = useState(0); // 0–1, higher = more likely real
  const [livenessPassed, setLivenessPassed] = useState(false); // simulate passing step 1

  const [opencvReady, setOpencvReady] = useState(false);
  const [cascadeLoaded, setCascadeLoaded] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null); // processing canvas (hidden)
  const overlayRef = useRef(null); // visible overlay for face boxes
  const streamRef = useRef(null);
  const prevGrayRef = useRef(null); // previous gray frame
  const faceClassifierRef = useRef(null); // cv.CascadeClassifier

  // Derived flag: is the classifier actually ready?
  const detectorReady = !!faceClassifierRef.current;

  const startCamera = async () => {
    try {
      setIsStarting(true);
      setStatus("Requesting camera access...");
      setFrameCount(0);
      setMotionLevel(0);
      setLivenessScore(0);
      setLivenessPassed(false);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Error: Camera API not supported in this browser.");
        setIsStarting(false);
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setStatus("Camera running. Ready for liveness checks.");
      } else {
        setStatus("Error: Video element not ready.");
      }
    } catch (err) {
      console.error(err);
      setStatus("Error: Could not access camera. Check permissions.");
    } finally {
      setIsStarting(false);
    }
  };

  // Cleanup on unmount: stop camera tracks and free OpenCV Mats / classifier
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (prevGrayRef.current) {
        prevGrayRef.current.delete();
        prevGrayRef.current = null;
      }
      if (faceClassifierRef.current) {
        faceClassifierRef.current.delete();
        faceClassifierRef.current = null;
      }
    };
  }, []);

  // 🔹 Initialize OpenCV + load face cascade
  useEffect(() => {
    let cancelled = false;

    const tryInit = async () => {
      if (cancelled) return;

      // Check if cv is loaded and filesystem APIs exist
      if (
        typeof cv === "undefined" ||
        !cv.FS_createDataFile ||
        !cv.CascadeClassifier
      ) {
        return; // opencv.js not fully ready yet
      }

      if (!opencvReady) {
        setOpencvReady(true);
      }

      // If cascade already loaded, nothing to do
      if (cascadeLoaded || faceClassifierRef.current) return;

      try {
        console.log(
          "Fetching cascade XML from /cascades/haarcascade_frontalface_default.xml"
        );

        // Fetch cascade XML from public folder
        const response = await fetch(
          "/cascades/haarcascade_frontalface_default.xml"
        );
        if (!response.ok) {
          console.error("Failed to fetch cascade XML", response.status);
          return;
        }

        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);

        const fileName = "haarcascade_frontalface_default.xml";
        const filePath = "/" + fileName; // use absolute path in OpenCV FS

        // Try to write file into OpenCV's in-memory filesystem.
        try {
          cv.FS_createDataFile("/", fileName, data, true, false, false);
          console.log("Cascade file written to OpenCV FS at", filePath);
        } catch (e) {
          console.log("Cascade file may already exist in FS:", e?.message);
        }

        // Create classifier and load the cascade
        const classifier = new cv.CascadeClassifier();
        const loaded = classifier.load(filePath);
        console.log("Classifier load result:", loaded);

        if (!loaded) {
          console.error("Failed to load face cascade from", filePath);
          classifier.delete();
          return;
        }

        faceClassifierRef.current = classifier;
        if (!cancelled) {
          setCascadeLoaded(true);
          console.log("Face cascade loaded successfully");
        }
      } catch (err) {
        console.error("Error loading cascade", err);
      }
    };

    const intervalId = setInterval(tryInit, 300);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [opencvReady, cascadeLoaded]);

  // Frame processing loop using OpenCV for motion detection + face boxes
  useEffect(() => {
    let intervalId;

    const processFrame = () => {
      // If we already passed liveness, no need to keep processing
      if (livenessPassed) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const overlay = overlayRef.current;
      if (!video || !canvas) return;

      // Only process if video has enough data
      if (video.readyState < 2) {
        return;
      }

      // Make sure OpenCV is loaded
      if (typeof cv === "undefined" || !cv.Mat) {
        return;
      }

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      if (width === 0 || height === 0) return;

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, width, height);

      // Read pixels into OpenCV Mat
      const src = cv.imread(canvas);
      const gray = new cv.Mat();

      // Convert RGBA -> grayscale
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      let motionScore = 0;

      if (prevGrayRef.current) {
        const diff = new cv.Mat();
        const thresh = new cv.Mat();

        // Absolute difference between current and previous gray frame
        cv.absdiff(gray, prevGrayRef.current, diff);

        // Threshold to ignore very small differences (noise)
        cv.threshold(diff, thresh, 25, 255, cv.THRESH_BINARY);

        // Count non-zero pixels = amount of motion
        const nonZero = cv.countNonZero(thresh);
        const totalPixels = thresh.rows * thresh.cols;

        motionScore = totalPixels > 0 ? nonZero / totalPixels : 0;

        // Cleanup mats
        diff.delete();
        thresh.delete();
      }

      // Replace previous gray frame
      if (prevGrayRef.current) {
        prevGrayRef.current.delete();
      }
      prevGrayRef.current = gray; // keep current gray for next frame

      // ---- Face detection + boxes (only if classifier exists) ----
      if (faceClassifierRef.current && overlay) {
        overlay.width = width;
        overlay.height = height;
        const octx = overlay.getContext("2d");
        octx.clearRect(0, 0, width, height);

        const faces = new cv.RectVector();

        try {
          // simpler call: no cv.Size to avoid weirdness
          faceClassifierRef.current.detectMultiScale(
            gray,
            faces,
            1.1,
            3,
            0
          );

          octx.lineWidth = 2;
          octx.strokeStyle = "rgba(74, 222, 128, 0.9)"; // Tailwind-ish green

          for (let i = 0; i < faces.size(); i++) {
            const face = faces.get(i);
            octx.strokeRect(face.x, face.y, face.width, face.height);
          }

          // console.log("Faces found:", faces.size());
        } catch (e) {
          console.error("Error during detectMultiScale:", e);
        } finally {
          faces.delete();
        }
      } else if (overlay) {
        // If no detector yet, just clear overlay
        overlay.width = width;
        overlay.height = height;
        const octx = overlay.getContext("2d");
        octx.clearRect(0, 0, width, height);
      }

      // Cleanup src (we keep gray in prevGrayRef)
      src.delete();

      // Update React state (lightweight)
      setFrameCount((prev) => prev + 1);
      setMotionLevel(motionScore);

      // Smooth liveness score over time using exponential moving average
      setLivenessScore((prev) => {
        const alpha = 0.25; // a bit more responsive
        const next = prev * (1 - alpha) + motionScore * alpha;
        return next;
      });
    };

    // Run ~5 times per second
    intervalId = setInterval(processFrame, 200);

    return () => {
      clearInterval(intervalId);
    };
  }, [livenessPassed, detectorReady]);

  // ==== Derived UI values (rescaled) ====

  // Raw motion in %
  const motionPercent = Math.round(motionLevel * 100);

  // Amplify the livenessScore so typical motion maps to ~60–80%
  const amplified = livenessScore * 5;
  const normalized = Math.max(0, Math.min(1, amplified));
  const livePercent = Math.round(normalized * 100);

  let livenessLabel = "Analyzing...";
  let livenessColor = "text-gray-300";
  let badgeText = "ANALYZING";
  let badgeBg = "bg-gray-800/80 border border-gray-600";

  if (frameCount > 10 && !livenessPassed) {
    if (livePercent < 20) {
      livenessLabel = "Very low motion — maybe fake / static";
      livenessColor = "text-red-500";
      badgeText = "MAYBE FAKE";
      badgeBg = "bg-red-600/80 border border-red-400";
    } else if (livePercent < 55) {
      livenessLabel = "Low / inconsistent motion — suspicious or very still";
      livenessColor = "text-yellow-400";
      badgeText = "SUSPICIOUS";
      badgeBg = "bg-yellow-500/80 border border-yellow-300";
    } else {
      livenessLabel = "Good motion — likely real person";
      livenessColor = "text-green-400";
      badgeText = "LIKELY REAL";
      badgeBg = "bg-green-600/80 border border-green-400";
    }
  }

  // ==== Decide when liveness "passes" and move to next step ====

  useEffect(() => {
    const PASS_THRESHOLD = 80; // % liveness needed
    const MIN_FRAMES = 50; // minimum frames processed before we trust it

    if (
      !livenessPassed &&
      frameCount > MIN_FRAMES &&
      livePercent >= PASS_THRESHOLD
    ) {
      // Mark as passed
      setLivenessPassed(true);
      setStatus("Liveness passed. Moving to next step...");

      // Stop camera stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    }
  }, [frameCount, livePercent, livenessPassed]);

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center p-4">
      <div className="w-full max-w-4xl space-y-4">
        <h1 className="text-3xl font-bold text-center mb-2">
          Liveness Check Demo
        </h1>

        <p className="text-gray-400 text-center text-sm mb-2">
          Step 1 in your multi-factor login flow: simple motion-based liveness
          using OpenCV, plus basic face detection. Not production-grade, but a
          good first signal.
        </p>

        {/* OpenCV / cascade status indicators */}
        <div className="flex justify-center gap-4 text-xs text-gray-400 mb-1">
          <span>
            OpenCV loaded:{" "}
            <span className={opencvReady ? "text-green-400" : "text-red-400"}>
              {opencvReady ? "Yes" : "No"}
            </span>
          </span>
          <span>
            Face detector:{" "}
            <span
              className={detectorReady ? "text-green-400" : "text-red-400"}
            >
              {detectorReady ? "Loaded" : "Not loaded"}
            </span>
          </span>
        </div>

        {!livenessPassed && (
          <div className="flex justify-center mb-2">
            <button
              onClick={startCamera}
              disabled={isStarting}
              className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 disabled:bg-red-900 text-sm font-medium transition"
            >
              {isStarting ? "Starting camera..." : "Start camera"}
            </button>
          </div>
        )}

        {/* If liveness not passed yet -> show camera + status.
            If passed -> show dummy "next factor" view. */}
        {!livenessPassed ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Video area */}
            <div className="md:col-span-2 bg-gray-900 rounded-xl border border-gray-800 aspect-video flex items-center justify-center overflow-hidden relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />

              {/* Hidden canvas used for processing frames with OpenCV */}
              <canvas ref={canvasRef} className="hidden" />

              {/* Visible overlay canvas for face boxes */}
              <canvas
                ref={overlayRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />

              {/* Liveness badge overlay */}
              <div className="absolute top-4 left-4">
                <div
                  className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold shadow-lg backdrop-blur-sm ${badgeBg}`}
                >
                  <span>{badgeText}</span>
                  <span className="text-white/80 font-mono">
                    {livePercent}%
                  </span>
                </div>
              </div>
            </div>

            {/* Status / Liveness area */}
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-col justify-between">
              <div>
                <h2 className="text-lg font-semibold mb-2">Status</h2>
                <p className="text-sm text-gray-300 mb-2">{status}</p>

                <p className="text-xs text-gray-400 mb-1">
                  Frames processed:{" "}
                  <span className="font-mono">{frameCount}</span>
                </p>

                <p className="text-xs text-gray-400 mb-1">
                  Motion level (raw):{" "}
                  <span className="font-mono">
                    {motionPercent}% (higher = more movement)
                  </span>
                </p>

                <div className="mt-3">
                  <p className="text-xs text-gray-400 mb-1">
                    Liveness estimate (scaled):
                  </p>
                  <p className={`text-base font-semibold ${livenessColor}`}>
                    {livePercent}% — {livenessLabel}
                  </p>
                </div>
              </div>

              <div className="mt-4 text-xs text-gray-500">
                Once your motion stays above the threshold for a short time,
                this step will be marked as passed and we&apos;ll move to the
                next factor.
              </div>
            </div>
          </div>
        ) : (
          // Dummy "next page" content (same route, different view)
          <div className="mt-4 bg-gray-900 rounded-xl border border-green-600 p-6 text-center space-y-3">
            <h2 className="text-2xl font-bold text-green-400">
              Liveness Passed ✅
            </h2>
            <p className="text-gray-300 text-sm max-w-md mx-auto">
              This simulates navigating to the next step in your multi-factor
              login flow.
            </p>
            <p className="text-gray-400 text-xs max-w-md mx-auto">
              In a real app, you would now render the next factor here (OTP,
              password, device confirmation, etc.). You can treat this as
              <span className="font-mono"> &quot;step 2&quot; </span> in your
              flow.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
