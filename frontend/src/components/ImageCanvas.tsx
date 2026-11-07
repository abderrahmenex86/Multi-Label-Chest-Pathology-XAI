import React from 'react';
import { ScanLine } from 'lucide-react';

interface ImageCanvasProps {
    preview: string | null;
    heatmap: string | null;
    opacity: number;
    target?: string;
}

export const ImageCanvas: React.FC<ImageCanvasProps> = ({
    preview,
    heatmap,
    opacity,
    target,
}) => {
    const hasImage = Boolean(preview);
    const hasHeatmap = Boolean(heatmap);

    return (
        <div className='flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-line bg-surface'>
            <div className='flex shrink-0 items-center justify-between gap-3 border-b border-line bg-white px-4 py-3'>
                <div className='flex min-w-0 items-center gap-2 text-xs text-muted'>
                    <span
                        className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                            hasHeatmap ? 'bg-brand' : 'bg-line'
                        }`}
                    />
                    <span className='truncate'>
                        {hasHeatmap ?
                            `HiResCAM: ${target || 'selected finding'}`
                        :   'Preview'}
                    </span>
                </div>
                {hasHeatmap && (
                    <span className='shrink-0 text-[11px] font-medium text-muted'>
                        Overlay {Math.round(opacity * 100)}%
                    </span>
                )}
            </div>

            <div className='scan-grid relative flex min-h-0 flex-1 items-center justify-center p-4 lg:p-6'>
                {hasImage ?
                    <div className='relative h-full w-full'>
                        <img
                            src={preview ?? ''}
                            alt='Chest X-ray'
                            className='absolute inset-0 h-full w-full rounded-xl object-contain'
                        />

                        {hasHeatmap && (
                            <img
                                src={heatmap ?? ''}
                                alt={
                                    target ?
                                        `HiResCAM explanation for ${target}`
                                    :   'HiResCAM explanation'
                                }
                                className='pointer-events-none absolute inset-0 h-full w-full rounded-xl object-contain transition-opacity duration-150'
                                style={{ opacity }}
                            />
                        )}
                    </div>
                :   <div className='flex max-w-[280px] flex-col items-center gap-3 text-center'>
                        <div className='flex h-12 w-12 items-center justify-center rounded-2xl border border-line bg-white text-brand'>
                            <ScanLine className='h-6 w-6' strokeWidth={1.6} />
                        </div>
                        <div>
                            <p className='font-heading text-sm font-medium text-ink'>
                                No X-ray selected
                            </p>
                            <p className='mt-1 text-xs leading-relaxed text-muted'>
                                Add a chest X-ray to see model findings and a
                                HiResCAM overlay.
                            </p>
                        </div>
                    </div>
                }
            </div>
        </div>
    );
};
