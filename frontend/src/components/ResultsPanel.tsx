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
    const filtered = predictions.filter(
        (item) => item.probability >= item.threshold
    );

    return (
        <div className='flex-1 min-h-0 flex flex-col gap-3 p-4 bg-slate-dark rounded-xl border border-steel-blue/40'>
            <div className='flex items-center justify-between pb-2 border-b border-steel-blue/30 shrink-0'>
                <h3 className='font-heading font-semibold text-xs text-pure-white uppercase tracking-wider'>
                    Pathology Findings
                </h3>
                <span className='text-xs font-mono text-silver-gray'>
                    {filtered.length} detected
                </span>
            </div>

            <div className='flex-1 overflow-y-auto pr-1 flex flex-col gap-2 min-h-0'>
                {predictions.length === 0 && (
                    <div className='flex flex-col items-center justify-center h-full text-silver-gray/60 gap-2 font-sans py-8'>
                        <ShieldAlert className='w-8 h-8 text-steel-blue/60' />
                        <p className='text-xs'>No analysis performed yet.</p>
                    </div>
                )}

                {predictions.length > 0 && filtered.length === 0 && (
                    <div className='py-8 text-center text-xs font-sans text-silver-gray'>
                        No active pathologies detected above threshold.
                    </div>
                )}

                {filtered.map((item) => {
                    const percentage = (item.probability * 100).toFixed(1);
                    const isHigh = item.probability > 0.7;
                    const isMedium =
                        item.probability >= 0.4 && item.probability <= 0.7;
                    const isSelected = target === item.name;

                    const barColor =
                        isHigh ? 'bg-coral-orange'
                        : isMedium ? 'bg-amber-500'
                        : 'bg-steel-blue';

                    const textColor =
                        isHigh ? 'text-coral-orange'
                        : isMedium ? 'text-amber-400'
                        : 'text-silver-gray';

                    return (
                        <div
                            key={item.name}
                            onClick={() => onSelectTarget(item.name)}
                            className={`flex flex-col gap-1.5 p-2.5 rounded-lg border transition-all cursor-pointer ${
                                isSelected ?
                                    'bg-steel-blue/20 border-coral-orange'
                                :   'bg-steel-blue/10 border-steel-blue/30 hover:border-steel-blue/60'
                            }`}>
                            <div className='flex justify-between items-center text-xs font-sans'>
                                <span className='font-semibold text-pure-white'>
                                    {item.name}
                                </span>
                                <span
                                    className={`font-mono font-bold ${textColor}`}>
                                    {percentage}%
                                </span>
                            </div>
                            <div className='w-full bg-steel-blue/30 rounded-full h-1.5 overflow-hidden'>
                                <div
                                    className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                                    style={{ width: `${percentage}%` }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className='pt-3 border-t border-steel-blue/30 flex flex-col gap-2 shrink-0'>
                <span className='font-heading font-semibold text-xs text-pure-white uppercase tracking-wider'>
                    Visualization
                </span>

                <div className='flex flex-col gap-1'>
                    <div className='flex justify-between items-center text-xs font-sans'>
                        <label className='text-silver-gray text-xs'>
                            Opacity
                        </label>
                        <span className='text-pure-white font-mono font-bold'>
                            {Math.round(opacity * 100)}%
                        </span>
                    </div>
                    <input
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
