import { useState, useRef } from "react";
import { predictSignature, type PredictionResponse } from "./api/predict";

type UploadSlot = "reference" | "test";

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
          ? "border-stone-200 cursor-default h-44"
          : "border-dashed h-44 cursor-pointer",
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
            className="w-full h-full object-contain p-3"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-200 flex items-center justify-center">
            <button
              onClick={(e) => { e.stopPropagation(); onFile(null); }}
              className="opacity-0 group-hover:opacity-100 transition-opacity text-white text-xs font-medium bg-black/60 hover:bg-black/80 rounded-lg px-3 py-1.5"
            >
              Remove
            </button>
          </div>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/40 to-transparent px-3 py-2">
            <p className="text-white text-xs truncate">{file.name}</p>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center gap-2 px-4 text-center select-none">
          <div className={[
            "w-9 h-9 rounded-xl flex items-center justify-center border transition-colors",
            dragging ? "border-stone-400 bg-stone-200" : "border-stone-200 bg-white",
          ].join(" ")}>
            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
              <path d="M10 3v10M10 3l-3 3M10 3l3 3" stroke="#78716c" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3 13v2a2 2 0 002 2h10a2 2 0 002-2v-2" stroke="#78716c" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-stone-700">{dragging ? "Release to upload" : label}</p>
            <p className="text-xs text-stone-400 mt-0.5">{sublabel}</p>
          </div>
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

const ConfidenceBadge = ({ confidence, label }: { confidence: number; label: string }) => {
  const pct = Math.round(confidence * 100);
  const isGenuine = label.toLowerCase().includes("genuine") || label.toLowerCase().includes("match");
  return (
    <div className={[
      "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold tracking-wide",
      isGenuine
        ? "bg-emerald-100 text-emerald-700"
        : "bg-red-100 text-red-600",
    ].join(" ")}>
      <span className={[
        "w-1.5 h-1.5 rounded-full",
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

  const handleSubmit = async () => {
    if (!referenceFile || !testFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictSignature(referenceFile, testFile);
      setResult(data);
    } catch {
      setError("Failed to connect to backend. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-50 flex items-center justify-center p-6">
      <div className="w-full max-w-md">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold tracking-widest text-stone-400 uppercase">Signature Verifier</span>
          </div>
          <h1 className="text-2xl font-semibold text-stone-900 mt-3">Verify a signature</h1>
          <p className="text-sm text-stone-400 mt-1">Upload a reference and test image to compare authenticity.</p>
        </div>

        {/* Upload slots */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <p className="text-sm font-medium text-stone-500 mb-1.5 uppercase tracking-wide">Reference</p>
            <FileSlot
              label="Drop reference"
              sublabel="The known genuine signature"
              file={referenceFile}
              onFile={setReferenceFile}
            />
          </div>
          <div>
            <p className="text-sm font-medium text-stone-500 mb-1.5 uppercase tracking-wide">Test</p>
            <FileSlot
              label="Drop test"
              sublabel="The signature to verify"
              file={testFile}
              onFile={setTestFile}
            />
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={[
            "w-full py-3 rounded-xl text-sm font-semibold transition-all duration-150",
            canSubmit
              ? "bg-stone-900 text-white hover:bg-stone-700 active:scale-[0.98]"
              : "bg-stone-200 text-stone-400 cursor-not-allowed",
          ].join(" ")}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="32" strokeDashoffset="12" strokeLinecap="round"/>
              </svg>
              Verifying…
            </span>
          ) : "Verify Signature"}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-3 flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3">
            <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor" className="shrink-0">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.75-5.25a.75.75 0 001.5 0v-4a.75.75 0 00-1.5 0v4zm.75-6.5a1 1 0 100 2 1 1 0 000-2z" clipRule="evenodd"/>
            </svg>
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-4 bg-white border border-stone-200 rounded-2xl overflow-hidden">
            <div className="px-5 py-4 border-b border-stone-100 flex items-center justify-between">
              <span className="text-sm font-semibold text-stone-800">Result</span>
              <ConfidenceBadge confidence={result.confidence} label={result.label} />
            </div>
            <div className="px-5 py-4 space-y-3">
              {/* Confidence bar */}
              <div>
                <div className="flex justify-between text-xs text-stone-400 mb-1.5">
                  <span>Confidence</span>
                  <span className="font-medium text-stone-600">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-stone-800 rounded-full transition-all duration-500"
                    style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
              {/* Stats */}
              <div className="grid grid-cols-2 gap-2 pt-1">
                <div className="bg-stone-50 rounded-xl px-3 py-2.5">
                  <p className="text-xs text-stone-400 mb-0.5">Distance</p>
                  <p className="text-sm font-semibold text-stone-800">{typeof result.distance === "number" ? result.distance.toFixed(4) : result.distance}</p>
                </div>
                <div className="bg-stone-50 rounded-xl px-3 py-2.5">
                  <p className="text-xs text-stone-400 mb-0.5">Verdict</p>
                  <p className="text-sm font-semibold text-stone-800">{result.label}</p>
                </div>
              </div>
              {/* Message */}
              <p className="text-xs text-stone-400 pt-0.5">{result.message}</p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default SignatureVerifier;