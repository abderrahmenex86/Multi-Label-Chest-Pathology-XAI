export interface Prediction {
    name: string;
    probability: number;
    threshold: number;
    positive: boolean;
}

export interface AnalysisPayload {
    predictions: Prediction[];
    heatmap: string;
    target: string;
    run: string;
    architecture: string;
}
