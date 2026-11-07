import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ImageCanvas } from './components/ImageCanvas';
import { ResultsPanel } from './components/ResultsPanel';
import { ActionIsland } from './components/ActionIsland';
import { analyzeImage, errorMessage } from './api';
import type { Prediction } from './types';

export const App: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [heatmap, setHeatmap] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [opacity, setOpacity] = useState<number>(0.5);
    const [target, setTarget] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const previewRef = useRef<string | null>(null);
    const requestRef = useRef<AbortController | null>(null);

    const revokePreview = useCallback(() => {
        if (previewRef.current) {
            URL.revokeObjectURL(previewRef.current);
            previewRef.current = null;
        }
    }, []);

    const handleFileSelect = (selectedFile: File) => {
        requestRef.current?.abort();
        requestRef.current = null;

        revokePreview();
        const nextPreview = URL.createObjectURL(selectedFile);
        previewRef.current = nextPreview;

        setFile(selectedFile);
        setPreview(nextPreview);
        setHeatmap(null);
        setPredictions([]);
        setTarget('');
        setError(null);
        setLoading(false);
    };

    const analyze = useCallback(
        async (selectedTarget?: string) => {
            if (!file) {
                return;
            }

            requestRef.current?.abort();
            const controller = new AbortController();
            requestRef.current = controller;

            setLoading(true);
            setError(null);

            try {
                const payload = await analyzeImage({
                    file,
                    target: selectedTarget,
                    signal: controller.signal,
                });

                if (requestRef.current !== controller) {
                    return;
                }

                setPredictions(payload.predictions);
                setHeatmap(`data:image/png;base64,${payload.heatmap}`);
                setTarget(payload.target);
            } catch (caught) {
                if (caught instanceof DOMException && caught.name === 'AbortError') {
                    return;
                }
                if (requestRef.current === controller) {
                    setError(errorMessage(caught));
                }
            } finally {
                if (requestRef.current === controller) {
                    setLoading(false);
                }
            }
        },
        [file]
    );

    const handleTargetChange = (newTarget: string) => {
        if (!file || loading) {
            return;
        }
        setTarget(newTarget);
        void analyze(newTarget);
    };

    useEffect(() => {
        return () => {
            requestRef.current?.abort();
            if (previewRef.current) {
                URL.revokeObjectURL(previewRef.current);
            }
        };
    }, []);

    return (
        <div className='min-h-dvh lg:h-dvh lg:max-h-dvh lg:overflow-hidden flex flex-col bg-slate-dark text-pure-white font-sans p-4 gap-3'>
            <header className='flex flex-row justify-between items-center pb-2 border-b border-steel-blue/30 shrink-0 gap-4'>
                <div>
                    <h1 className='font-heading text-lg font-bold text-pure-white'>
                        Chest Pathology Visualizer
                    </h1>
                    <p className='text-xs text-silver-gray'>
                        Deep Learning &amp; HiResCAM Explainability Research
                        Workspace
                    </p>
                </div>
                <span className='hidden sm:inline-flex shrink-0 items-center rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-[10px] font-heading font-semibold uppercase tracking-wider text-amber-300'>
                    Research use only
                </span>
            </header>

            {error && (
                <div
                    role='alert'
                    className='shrink-0 rounded-lg border border-red-800/60 bg-red-950/60 px-3 py-2 text-xs text-red-100'>
                    {error}
                </div>
            )}

            <div className='flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 lg:min-h-0'>
                <div className='lg:col-span-9 flex flex-col min-h-[50vh] lg:min-h-0'>
                    <ImageCanvas
                        preview={preview}
                        heatmap={heatmap}
                        opacity={opacity}
                        target={target}
                    />
                </div>

                <div className='lg:col-span-3 flex flex-col gap-3 lg:min-h-0'>
                    <ResultsPanel
                        predictions={predictions}
                        opacity={opacity}
                        setOpacity={setOpacity}
                        target={target}
                        onSelectTarget={handleTargetChange}
                        disabled={loading}
                    />

                    <ActionIsland
                        onFileSelect={handleFileSelect}
                        selectedFile={file}
                        onSubmit={() => void analyze(target || undefined)}
                        loading={loading}
                    />
                </div>
            </div>

            <footer className='pt-2 border-t border-steel-blue/30 flex flex-wrap gap-2 items-center justify-between text-xs font-sans text-silver-gray shrink-0'>
                <span>
                    Research prototype. Not a medical device and not for clinical
                    use.
                </span>
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
