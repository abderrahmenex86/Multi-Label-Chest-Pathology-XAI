import React, { useState } from 'react';
import { ImageCanvas } from './components/ImageCanvas';
import { ResultsPanel } from './components/ResultsPanel';
import { ActionIsland } from './components/ActionIsland';
import type { Prediction, AnalysisPayload } from './types';

export const App: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [heatmap, setHeatmap] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [opacity, setOpacity] = useState<number>(0.5);
    const [target, setTarget] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);

    const handleFileSelect = (selectedFile: File) => {
        setFile(selectedFile);
        setPreview(URL.createObjectURL(selectedFile));
        setHeatmap(null);
        setPredictions([]);
        setTarget('');
    };

    const analyze = async (selectedTarget?: string) => {
        if (!file) return;
        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);
        if (selectedTarget) {
            formData.append('target', selectedTarget);
        }

        const host = import.meta.env.VITE_API_URL || 'http://localhost:8000';

        try {
            const response = await fetch(`${host}/api/v1/predict`, {
                method: 'POST',
                body: formData,
            });

            const data: AnalysisPayload = await response.json();
            setPredictions(data.predictions);
            setHeatmap(`data:image/png;base64,${data.heatmap}`);
        } catch (error) {
            console.error('Analysis request failed', error);
        } finally {
            setLoading(false);
        }
    };

    const handleTargetChange = (newTarget: string) => {
        setTarget(newTarget);
        if (file && predictions.length > 0) {
            analyze(newTarget);
        }
    };

    return (
        <div className='h-screen max-h-screen overflow-hidden flex flex-col bg-slate-dark text-pure-white font-sans p-4 gap-3'>
            <header className='flex flex-row justify-between items-center pb-2 border-b border-steel-blue/30 shrink-0'>
                <div>
                    <h1 className='font-heading text-lg font-bold text-pure-white'>
                        Chest Pathology Visualizer
                    </h1>
                    <p className='text-xs text-silver-gray'>
                        Deep Learning & HiResCAM Explainability Diagnostic
                        Workspace
                    </p>
                </div>
            </header>

            <div className='flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-0 overflow-hidden'>
                <div className='lg:col-span-9 flex flex-col min-h-0 h-full'>
                    <ImageCanvas
                        preview={preview}
                        heatmap={heatmap}
                        opacity={opacity}
                    />
                </div>

                <div className='lg:col-span-3 flex flex-col gap-3 min-h-0 h-full'>
                    <ResultsPanel
                        predictions={predictions}
                        opacity={opacity}
                        setOpacity={setOpacity}
                        target={target}
                        onSelectTarget={handleTargetChange}
                        disabled={loading || !heatmap}
                    />

                    <ActionIsland
                        onFileSelect={handleFileSelect}
                        selectedFile={file}
                        onSubmit={() => analyze(target)}
                        loading={loading}
                    />
                </div>
            </div>

            <footer className='pt-2 border-t border-steel-blue/30 flex items-center justify-between text-xs font-sans text-silver-gray shrink-0'>
                <span>made with love, by humans.</span>
                <a
                    href='https://github.com/abderrahmenex86'
                    target='_blank'
                    rel='noopener noreferrer'
                    className='flex items-center gap-1.5 text-coral-orange hover:underline font-mono'>
                    <svg
                        className='w-3.5 h-3.5 fill-current'
                        viewBox='0 0 24 24'>
                        <path d='M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z' />
                    </svg>
                    <span>@github.com/abderrahmenex86</span>
                </a>
            </footer>
        </div>
    );
};

export default App;
