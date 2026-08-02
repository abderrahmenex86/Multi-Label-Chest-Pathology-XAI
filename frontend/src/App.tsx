import React, { useState } from 'react';
import { Activity, Sparkles, Github } from 'lucide-react';
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
    const [threshold, setThreshold] = useState<number>(0.2);
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
                <div className='flex items-center gap-3'>
                    <div className='p-2 bg-coral-orange rounded-xl text-pure-white shadow-md'>
                        <Activity className='w-5 h-5' />
                    </div>
                    <div>
                        <h1 className='font-heading text-lg font-bold text-pure-white'>
                            Chest Pathology Visualizer
                        </h1>
                        <p className='text-xs text-silver-gray'>
                            Deep Learning & HiResCAM Explainability Diagnostic
                            Workspace
                        </p>
                    </div>
                </div>
                <div className='flex items-center gap-2 px-3 py-1 bg-steel-blue/20 border border-steel-blue/40 rounded-full text-xs font-mono text-coral-orange'>
                    <Sparkles className='w-3.5 h-3.5' />
                    <span>DenseNet / ConvNeXt</span>
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
                        threshold={threshold}
                        setThreshold={setThreshold}
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
                    <Github className='w-3.5 h-3.5' />
                    <span>@github.com/abderrahmenex86</span>
                </a>
            </footer>
        </div>
    );
};

export default App;
