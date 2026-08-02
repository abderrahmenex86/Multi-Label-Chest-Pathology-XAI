import React, { useState } from 'react';
import { Activity, Sparkles, RefreshCw, Play } from 'lucide-react';
import { FileDropzone } from './components/FileDropzone';
import { ImageCanvas } from './components/ImageCanvas';
import { Controls } from './components/Controls';
import { ResultsPanel } from './components/ResultsPanel';
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
        <div className='h-screen max-h-screen overflow-hidden flex flex-col bg-ink-black-950 font-sans p-4 gap-3'>
            <header className='flex flex-row justify-between items-center pb-2 shrink-0'>
                <div className='flex items-center gap-3'>
                    <div>
                        <h1 className='font-heading text-lg font-bold text-school-bus-yellow-400'>
                            Multi Label Chest Pathology Visualizer with XAI
                        </h1>
                        <p className='text-xs text-school-bus-yellow-200'>
                            Deep Learning & HiResCAM Explainability Diagnostic
                            Workspace
                        </p>
                    </div>
                </div>
            </header>

            <div className='flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-0 overflow-hidden'>
                <div className='lg:col-span-8 flex flex-col gap-3 min-h-0 h-full'>
                    <ImageCanvas
                        preview={preview}
                        heatmap={heatmap}
                        opacity={opacity}
                    />

                    <div className='flex items-center gap-3 h-14 shrink-0'>
                        <FileDropzone
                            onFileSelect={handleFileSelect}
                            selectedFile={file}
                        />

                        <div className='flex-1 h-14 flex items-center'>
                            <button
                                onClick={() => analyze(target)}
                                disabled={loading || !file}
                                className='w-full h-full cursor-pointer bg-prussian-blue-600 hover:bg-prussian-blue-500 disabled:bg-ink-black-950 disabled:text-ink-black-400 disabled:border-ink-black-800 disabled:border text-white font-heading font-semibold text-xs rounded-xl transition-colors flex items-center justify-center gap-2 px-4'>
                                {loading ?
                                    <>
                                        <RefreshCw className='w-4 h-4 animate-spin text-prussian-blue-400' />
                                        <span>
                                            Analyzing & Generating Heatmap...
                                        </span>
                                    </>
                                :   <>
                                        <Play className='w-4 h-4 fill-current' />
                                        <span>
                                            {file ?
                                                'Run Pathology Analysis'
                                            :   'Upload Image First'}
                                        </span>
                                    </>
                                }
                            </button>
                        </div>
                    </div>
                </div>

                <div className='lg:col-span-4 flex flex-col gap-3 min-h-0 h-full'>
                    <ResultsPanel
                        predictions={predictions}
                        target={target}
                        onSelectTarget={handleTargetChange}
                    />

                    <Controls
                        opacity={opacity}
                        setOpacity={setOpacity}
                        target={target}
                        predictions={predictions}
                        onTargetChange={handleTargetChange}
                        disabled={loading || !heatmap}
                    />
                </div>
            </div>
        </div>
    );
};

export default App;
