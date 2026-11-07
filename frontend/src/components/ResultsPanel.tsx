import React from 'react';
import { ShieldAlert } from 'lucide-react';
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
        <div className='flex-1 min-h-0 flex flex-col gap-3 p-4 bg-slate-dark rounded-xl border border-steel-blue/40'>
            <div className='flex items-center justify-between pb-2 border-b border-steel-blue/30 shrink-0'>
                <h2 className='font-heading font-semibold text-xs text-pure-white uppercase tracking-wider'>
                    Pathology Findings
                </h2>
                <span className='text-xs font-mono text-silver-gray'>
                    {positiveCount} above threshold
                </span>
            </div>

            <div
                className='flex-1 overflow-y-auto pr-1 flex flex-col gap-2 min-h-0'
                aria-live='polite'>
                {predictions.length === 0 && (
                    <div className='flex flex-col items-center justify-center h-full text-silver-gray/70 gap-2 font-sans py-8'>
                        <ShieldAlert className='w-8 h-8 text-steel-blue/70' />
                        <p className='text-xs'>No analysis performed yet.</p>
                    </div>
                )}

                {predictions.length > 0 && positiveCount === 0 && (
                    <p className='py-2 text-center text-xs font-sans text-silver-gray'>
                        No findings are above their tuned threshold. You can
                        still select a finding to inspect its explanation.
                    </p>
                )}

                {predictions.map((item) => {
                    const percentage = Math.min(
                        100,
                        Math.max(0, item.probability * 100)
                    );
                    const isSelected = target === item.name;
                    const isPositive = item.probability >= item.threshold;

                    const barColor = isPositive ? 'bg-coral-orange' : 'bg-steel-blue';
                    const textColor = isPositive
                        ? 'text-coral-orange'
                        : 'text-silver-gray';

                    return (
                        <button
                            key={item.name}
                            type='button'
                            onClick={() => onSelectTarget(item.name)}
                            disabled={disabled}
                            aria-pressed={isSelected}
                            className={`w-full flex flex-col gap-1.5 p-2.5 rounded-lg border text-left transition-all enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-60 ${
                                isSelected
                                    ? 'bg-steel-blue/20 border-coral-orange'
                                    : 'bg-steel-blue/10 border-steel-blue/30 enabled:hover:border-steel-blue/60'
                            }`}>
                            <div className='flex justify-between items-center gap-2 text-xs font-sans'>
                                <span className='font-semibold text-pure-white'>
                                    {item.name}
                                </span>
                                <span
                                    className={`font-mono font-bold ${textColor}`}>
                                    {percentage.toFixed(1)}%
                                </span>
                            </div>
                            <div className='w-full bg-steel-blue/30 rounded-full h-1.5 overflow-hidden'>
                                <div
                                    className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                                    style={{ width: `${percentage}%` }}
                                />
                            </div>
                            <div className='flex justify-between items-center text-[10px] font-mono text-silver-gray/80'>
                                <span>
                                    threshold{' '}
                                    {(item.threshold * 100).toFixed(0)}%
                                </span>
                                <span>
                                    {isPositive ? 'detected' : 'below threshold'}
                                </span>
                            </div>
                        </button>
                    );
                })}
            </div>

            <div className='pt-3 border-t border-steel-blue/30 flex flex-col gap-2 shrink-0'>
                <span className='font-heading font-semibold text-xs text-pure-white uppercase tracking-wider'>
                    Visualization
                </span>

                <div className='flex flex-col gap-1'>
                    <div className='flex justify-between items-center text-xs font-sans'>
                        <label
                            htmlFor='heatmap-opacity'
                            className='text-silver-gray text-xs'>
                            Opacity
                        </label>
                        <span className='text-pure-white font-mono font-bold'>
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
                        className='w-full h-1.5 bg-steel-blue/30 rounded-lg appearance-none cursor-pointer accent-coral-orange disabled:opacity-40'
                    />
                </div>
            </div>
        </div>
    );
};
