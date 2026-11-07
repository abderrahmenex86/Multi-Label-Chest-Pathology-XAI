import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ScanLine } from 'lucide-react';
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
        <div className='flex min-h-dvh flex-col bg-white font-sans text-ink lg:h-dvh lg:max-h-dvh lg:overflow-hidden'>
            <header className='shrink-0 border-b border-line bg-white'>
                <div className='mx-auto flex w-full max-w-[1600px] items-center justify-between gap-6 px-4 py-4 lg:px-8'>
                    <div className='flex items-center gap-3'>
                        <div className='flex h-10 w-10 items-center justify-center rounded-xl bg-brand text-white shadow-sm'>
                            <ScanLine className='h-5 w-5' strokeWidth={2} />
                        </div>
                        <div>
                            <h1 className='font-heading text-[15px] font-semibold tracking-[-0.01em] text-ink'>
                                Chest X-Ray Studio
                            </h1>
                            <p className='text-xs text-muted'>
                                Multi-label pathology and HiResCAM
                            </p>
                        </div>
                    </div>

                    <span className='hidden items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-[11px] font-medium text-muted sm:inline-flex'>
                        <span className='h-1.5 w-1.5 rounded-full bg-brand' />
                        Research prototype
                    </span>
                </div>
            </header>

            {error && (
                <div className='mx-auto w-full max-w-[1600px] px-4 pt-4 lg:px-8'>
                    <div
                        role='alert'
                        className='rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800'>
                        {error}
                    </div>
                </div>
            )}

            <main className='mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-6 px-4 py-4 lg:min-h-0 lg:flex-row lg:gap-0 lg:px-8 lg:py-6'>
                <section className='flex min-h-[52vh] flex-1 flex-col lg:min-h-0 lg:pr-8'>
                    <ImageCanvas
                        preview={preview}
                        heatmap={heatmap}
                        opacity={opacity}
                        target={target}
                    />
                </section>

                <aside className='flex w-full flex-col gap-6 lg:min-h-0 lg:w-[360px] lg:border-l lg:border-line lg:pl-8 xl:w-[400px]'>
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
                </aside>
            </main>

            <footer className='shrink-0 border-t border-line bg-white'>
                <div className='mx-auto flex w-full max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-3 text-xs text-muted lg:px-8'>
                    <span>
                        Research prototype. Not a medical device and not for
                        clinical use.
                    </span>
                    <a
                        href='https://github.com/abderrahmenex86'
                        target='_blank'
                        rel='noopener noreferrer'
                        className='font-medium text-brand transition-colors hover:text-brand-dark'>
                        @github.com/abderrahmenex86
                    </a>
                </div>
            </footer>
        </div>
    );
};

export default App;
