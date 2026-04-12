import api from "./api";

export interface PredictionResponse {
  label: string;           // "genuine" or "forged"
  confidence: number;      // 0.0 - 1.0
  distance: number;        // raw Euclidean distance
  score: number;           // inverted distance
  reference_filename: string;
  test_filename: string;
  message: string;
}

export const predictSignature = async (
  referenceFile: File,
  testFile: File
): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append("reference", referenceFile);  // must match FastAPI param name
  formData.append("test", testFile);            // must match FastAPI param name

  const response = await api.post<PredictionResponse>("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};