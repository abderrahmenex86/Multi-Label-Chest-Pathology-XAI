import React from 'react';
import { ListFilter } from 'lucide-react';
import type { Prediction } from '../types';

interface ResultsPanelProps {
    predictions: Prediction[];
    opacity: number;
    setOpacity: (val: number) => void;
    target: string;
    onSelectTarget: (name: string) => void;
    disabled: boolean;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
    predictions,
    opacity,
    setOpacity,
    target,
    onSelectTarget,
    disabled,
}) => {
    const positiveCount = predictions.filter(
        (item) => item.probability >= item.threshold
    ).length;

    return (
        <section className='flex min-h-0 flex-1 flex-col'>
            <div className='flex items-end justify-between gap-4 pb-3'>
                <div>
                    <h2 className='font-heading text-sm font-semibold text-ink'>
                        Findings
                    </h2>
                    <p className='mt-1 text-xs text-muted'>
                        Select a finding to update the heatmap.
                    </p>
                </div>
                <span className='shrink-0 pb-0.5 text-[11px] font-medium text-muted'>
                    {positiveCount} above threshold
                </span>
            </div>

            <div
                className='min-h-0 flex-1 overflow-y-auto'
                aria-live='polite'>
                {predictions.length === 0 && (
                    <div className='flex h-full min-h-[180px] flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line bg-surface px-6 py-8 text-center'>
                        <ListFilter className='h-5 w-5 text-muted' strokeWidth={1.6} />
                        <div>
                            <p className='font-heading text-sm font-medium text-ink'>
                                Findings will appear here
                            </p>
                            <p className='mt-1 text-xs leading-relaxed text-muted'>
                                Analyze an X-ray to see all 14 pathology
                                probabilities.
                            </p>
                        </div>
                    </div>
                )}

                {predictions.length > 0 && positiveCount === 0 && (
                    <p className='mb-2 rounded-lg bg-surface px-3 py-2 text-xs leading-relaxed text-muted'>
                        No findings are above their tuned threshold. You can
                        still select one to inspect its explanation.
                    </p>
                )}

                {predictions.length > 0 && (
                    <div className='divide-y divide-line'>
                        {predictions.map((item) => {
                            const percentage = Math.min(
                                100,
                                Math.max(0, item.probability * 100)
                            );
                            const isSelected = target === item.name;
                            const isPositive =
                                item.probability >= item.threshold;

                            return (
                                <button
                                    key={item.name}
                                    type='button'
                                    onClick={() => onSelectTarget(item.name)}
                                    disabled={disabled}
                                    aria-pressed={isSelected}
                                    className={`w-full px-2 py-3 text-left transition-colors enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-60 ${
                                        isSelected
                                            ? 'bg-brand-soft'
                                            : 'enabled:hover:bg-surface'
                                    }`}>
                                    <div className='flex items-center justify-between gap-3'>
                                        <span className='flex min-w-0 items-center gap-2'>
                                            <span
                                                className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                                                    isPositive
                                                        ? 'bg-brand'
                                                        : 'bg-line'
                                                }`}
                                            />
                                            <span className='truncate text-sm font-medium text-ink'>
                                                {item.name}
                                            </span>
                                        </span>
                                        <span
                                            className={`shrink-0 text-sm font-semibold tabular-nums ${
                                                isPositive
                                                    ? 'text-brand'
                                                    : 'text-muted'
                                            }`}>
                                            {percentage.toFixed(1)}%
                                        </span>
                                    </div>

                                    <div className='mt-2 h-1 w-full overflow-hidden rounded-full bg-line'>
                                        <div
                                            className={`h-full rounded-full transition-all duration-300 ${
                                                isPositive
                                                    ? 'bg-brand'
                                                    : 'bg-muted/30'
                                            }`}
                                            style={{ width: `${percentage}%` }}
                                        />
                                    </div>

                                    <div className='mt-1.5 flex items-center justify-between gap-3 text-[11px] text-muted'>
                                        <span>
                                            {isPositive
                                                ? 'Detected'
                                                : 'Below threshold'}
                                        </span>
                                        <span>
                                            Threshold{' '}
                                            {Math.round(item.threshold * 100)}%
                                        </span>
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                )}
            </div>

            <div className='mt-4 shrink-0 border-t border-line pt-4'>
                <div className='flex items-center justify-between gap-4'>
                    <label
                        htmlFor='heatmap-opacity'
                        className='text-xs font-medium text-ink'>
                        Overlay opacity
                    </label>
                    <span className='text-xs font-medium tabular-nums text-muted'>
                        {Math.round(opacity * 100)}%
                    </span>
                </div>
                <input
                    id='heatmap-opacity'
                    type='range'
                    min='0'
                    max='1'
                    step='0.02'
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                    disabled={disabled}
                    className='mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-line accent-brand disabled:opacity-40'
                />
            </div>
        </section>
    );
};
