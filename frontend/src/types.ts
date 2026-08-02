export interface Prediction {
    name: string;
    probability: number;
    threshold: number;
}

export interface AnalysisPayload {
    predictions: Prediction[];
    heatmap: string;
}
