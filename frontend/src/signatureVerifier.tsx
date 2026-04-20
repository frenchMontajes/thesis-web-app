import { useState, useRef } from "react";
import { predictSignature, type PredictionResponse } from "./api/predict";
import { getErrorMessage } from "./utils/errorHandler";

interface FileSlotProps {
  label: string;
  sublabel: string;
  file: File | null;
  onFile: (f: File | null) => void;
}

const FileSlot = ({ label, sublabel, file, onFile }: FileSlotProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  return (
    <div
      onClick={() => !file && inputRef.current?.click()}
      onDragEnter={(e) => { e.preventDefault(); dragCounter.current++; setDragging(true); }}
      onDragLeave={() => { dragCounter.current--; if (dragCounter.current === 0) setDragging(false); }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      className={[
        "relative flex flex-col items-center justify-center rounded-2xl border transition-all duration-200 overflow-hidden group",
        file
          ? "border-stone-200 cursor-default h-64"
          : "border-dashed h-64 cursor-pointer",
        dragging && !file
          ? "border-stone-400 bg-stone-100"
          : !file
          ? "border-stone-300 bg-stone-50 hover:border-stone-400 hover:bg-stone-100"
          : "border-stone-200",
      ].join(" ")}
    >
      {file ? (
        <>
          <img
            src={URL.createObjectURL(file)}
            alt={label}
            className="w-full h-full object-contain p-4"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-200 flex items-center justify-center">
            <button
              onClick={(e) => { e.stopPropagation(); onFile(null); }}
              className="opacity-0 group-hover:opacity-100 transition-opacity text-white text-sm font-medium bg-black/60 hover:bg-black/80 rounded-lg px-4 py-2"
            >
              Remove
            </button>
          </div>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/40 to-transparent px-4 py-3">
            <p className="text-white text-sm truncate">{file.name}</p>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center gap-4 px-6 text-center select-none">
          <div className={[
            "w-14 h-14 rounded-2xl flex items-center justify-center border-2 transition-colors",
            dragging ? "border-stone-400 bg-stone-200" : "border-stone-200 bg-white",
          ].join(" ")}>
            <svg width="26" height="26" viewBox="0 0 20 20" fill="none">
              <path d="M10 3v10M10 3l-3 3M10 3l3 3" stroke="#78716c" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3 13v2a2 2 0 002 2h10a2 2 0 002-2v-2" stroke="#78716c" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <p className="text-base font-semibold text-stone-700">
              {dragging ? "Release to upload" : label}
            </p>
            <p className="text-sm text-stone-400 mt-1">{sublabel}</p>
          </div>
          <span className="text-sm text-stone-400">
            or{" "}
            <button
              onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
              className="text-stone-600 underline underline-offset-2 hover:text-stone-900 transition-colors font-medium"
            >
              browse files
            </button>
          </span>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }}
      />
    </div>
  );
};

const DistanceBadge = ({ distance, label }: { distance: number; label: string }) => {
  const isGenuine = label.toLowerCase().includes("genuine") || label.toLowerCase().includes("match");
  return (
    <div className={[
      "inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold tracking-wide",
      isGenuine
        ? "bg-emerald-100 text-emerald-700"
        : "bg-red-100 text-red-600",
    ].join(" ")}>
      <span className={[
        "w-2 h-2 rounded-full",
        isGenuine ? "bg-emerald-500" : "bg-red-500",
      ].join(" ")} />
      {label.toUpperCase()}
    </div>
  );
};

const SignatureVerifier = () => {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [testFile, setTestFile]           = useState<File | null>(null);
  const [result, setResult]               = useState<PredictionResponse | null>(null);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState<string | null>(null);

  const canSubmit = referenceFile && testFile && !loading;

  // Clears result + error whenever a slot is set to null (file removed)
  const handleFileChange = (
    setter: React.Dispatch<React.SetStateAction<File | null>>,
    file: File | null
  ) => {
    setter(file);
    if (file === null) {
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!referenceFile || !testFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictSignature(referenceFile, testFile);
      setResult(data);
    } catch (err: unknown) {
      console.error("🔥 UI ERROR:", err);
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-50 flex items-center justify-center p-8">
      <div className="w-full max-w-xl">

        {/* Header */}
        <div className="mb-10">
          <span className="text-xs font-bold tracking-widest text-stone-400 uppercase">
            Signature Verifier
          </span>
          <h1 className="text-3xl font-semibold text-stone-900 mt-3">Verify a signature</h1>
          <p className="text-base text-stone-400 mt-2">
            Upload a reference and test image to compare authenticity.
          </p>
        </div>

        {/* Upload slots */}
        <div className="grid grid-cols-2 gap-4 mb-5">
          <div>
            <p className="text-sm font-semibold text-stone-500 mb-2 uppercase tracking-wider">Reference</p>
            <FileSlot
              label="Drop reference"
              sublabel="The known genuine signature"
              file={referenceFile}
              onFile={(f) => handleFileChange(setReferenceFile, f)}
            />
          </div>
          <div>
            <p className="text-sm font-semibold text-stone-500 mb-2 uppercase tracking-wider">Test</p>
            <FileSlot
              label="Drop test"
              sublabel="The signature to verify"
              file={testFile}
              onFile={(f) => handleFileChange(setTestFile, f)}
            />
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={[
            "w-full py-4 rounded-xl text-base font-semibold transition-all duration-150",
            canSubmit
              ? "bg-stone-900 text-white hover:bg-stone-700 active:scale-[0.98]"
              : "bg-stone-200 text-stone-400 cursor-not-allowed",
          ].join(" ")}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="32" strokeDashoffset="12" strokeLinecap="round"/>
              </svg>
              Verifying…
            </span>
          ) : "Verify Signature"}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-4 flex items-center gap-2.5 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-5 py-4">
            <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor" className="shrink-0">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.75-5.25a.75.75 0 001.5 0v-4a.75.75 0 00-1.5 0v4zm.75-6.5a1 1 0 100 2 1 1 0 000-2z" clipRule="evenodd"/>
            </svg>
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-5 bg-white border border-stone-200 rounded-2xl overflow-hidden">
            <div className="px-6 py-5 border-b border-stone-100 flex items-center justify-between">
              <span className="text-base font-semibold text-stone-800">Result</span>
              <DistanceBadge distance={result.distance} label={result.label} />
            </div>
            <div className="px-6 py-5 space-y-4">
              {/* Distance bar */}
              <div>
                <div className="flex justify-between text-sm text-stone-400 mb-2">
                  <span>Distance</span>
                  <span className="font-semibold text-stone-700">{(result.distance).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-stone-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-stone-800 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(result.distance * 100, 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
              {/* Stats */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="bg-stone-50 rounded-xl px-4 py-3">
                  <p className="text-xs text-stone-400 mb-1">Distance</p>
                  <p className="text-base font-semibold text-stone-800">
                    {typeof result.distance === "number" ? result.distance.toFixed(4) : result.distance}
                  </p>
                </div>
                <div className="bg-stone-50 rounded-xl px-4 py-3">
                  <p className="text-xs text-stone-400 mb-1">Verdict</p>
                  <p className="text-base font-semibold text-stone-800">{result.label}</p>
                </div>
              </div>
              {/* Message */}
              <p className="text-sm text-stone-400 pt-1">{result.message}</p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default SignatureVerifier;