import { isAxiosError } from "axios";

export function getErrorMessage(err: unknown): string {
  // Axios error (backend / network)
  if (isAxiosError(err)) {
    if (err.response) {
      const data = err.response.data as { detail?: string };

      return (
        data?.detail ||
        JSON.stringify(err.response.data) ||
        "Server error occurred"
      );
    }

    if (err.request) {
      return "Backend is not running or unreachable.";
    }

    return err.message;
  }

  // Native JS error
  if (err instanceof Error) {
    return err.message;
  }

  return "Unknown error occurred";
}