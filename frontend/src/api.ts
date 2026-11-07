import type { AnalysisPayload } from './types';

const DEFAULT_HOST = 'http://localhost:8000';

function apiHost(): string {
    return (import.meta.env.VITE_API_URL || DEFAULT_HOST).replace(/\/$/, '');
}

interface AnalyzeOptions {
    file: File;
    target?: string;
    signal?: AbortSignal;
}

export async function analyzeImage({
    file,
    target,
    signal,
}: AnalyzeOptions): Promise<AnalysisPayload> {
    const formData = new FormData();
    formData.append('file', file);
    if (target) {
        formData.append('target', target);
    }

    const response = await fetch(`${apiHost()}/api/v1/predict`, {
        method: 'POST',
        body: formData,
        signal,
    });

    if (!response.ok) {
        let detail = `Request failed (${response.status} ${response.statusText})`;
        try {
            const body: unknown = await response.json();
            if (
                body &&
                typeof body === 'object' &&
                'detail' in body &&
                typeof body.detail === 'string'
            ) {
                detail = body.detail;
            }
        } catch {
            // The response was not JSON; keep the status-based message.
        }
        throw new Error(detail);
    }

    const body: unknown = await response.json();
    if (!isAnalysisPayload(body)) {
        throw new Error('The server returned an invalid prediction payload.');
    }
    return body;
}

function isAnalysisPayload(value: unknown): value is AnalysisPayload {
    if (!value || typeof value !== 'object') {
        return false;
    }

    const payload = value as Record<string, unknown>;
    if (
        typeof payload.heatmap !== 'string' ||
        typeof payload.target !== 'string' ||
        typeof payload.run !== 'string' ||
        typeof payload.architecture !== 'string' ||
        !Array.isArray(payload.predictions)
    ) {
        return false;
    }

    return payload.predictions.every((item) => {
        if (!item || typeof item !== 'object') {
            return false;
        }
        const prediction = item as Record<string, unknown>;
        return (
            typeof prediction.name === 'string' &&
            typeof prediction.probability === 'number' &&
            typeof prediction.threshold === 'number' &&
            typeof prediction.positive === 'boolean'
        );
    });
}

export function errorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return 'Analysis failed. Is the backend running?';
}
