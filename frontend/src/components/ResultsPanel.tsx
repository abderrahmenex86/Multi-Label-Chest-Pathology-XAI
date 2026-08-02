import React from 'react';
import { Activity, ShieldAlert } from 'lucide-react';
import type { Prediction } from '../types';

interface ResultsPanelProps {
    predictions: Prediction[];
    target: string;
    onSelectTarget: (name: string) => void;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
    predictions,
    target,
    onSelectTarget,
}) => {
    const filtered = predictions.filter(
        (item) => item.probability >= item.threshold
    );

    return (
        <div className='flex-1 min-h-0 flex flex-col gap-3 p-4 bg-ink-black-950 rounded-xl border border-ink-black-800'>
            <div className='flex items-center justify-between pb-2 border-b border-ink-black-800 shrink-0'>
                <div className='flex items-center gap-2 text-prussian-blue-400'>
                    <Activity className='w-4 h-4' />
                    <h3 className='font-heading font-semibold text-xs text-ink-black-50 uppercase tracking-wider'>
                        Pathology Findings
                    </h3>
                </div>
                <span className='text-xs font-mono text-ink-black-300'>
                    {filtered.length} detected
                </span>
            </div>

            <div className='flex-1 overflow-y-auto pr-1 flex flex-col gap-2 min-h-0'>
                {predictions.length === 0 && (
                    <div className='flex flex-col items-center justify-center h-full text-ink-black-400 gap-2 font-sans py-8'>
                        <ShieldAlert className='w-8 h-8 text-ink-black-700' />
                        <p className='text-xs'>No analysis performed yet.</p>
                    </div>
                )}

                {predictions.length > 0 && filtered.length === 0 && (
                    <div className='py-8 text-center text-xs font-sans text-ink-black-400'>
                        No active pathologies detected above optimal thresholds.
                    </div>
                )}

                {filtered.map((item) => {
                    const percentage = (item.probability * 100).toFixed(1);
                    const isHigh = item.probability > 0.7;
                    const isMedium =
                        item.probability >= 0.4 && item.probability <= 0.7;
                    const isSelected = target === item.name;

                    const barColor =
                        isHigh ? 'bg-red-500'
                        : isMedium ? 'bg-amber-500'
                        : 'bg-slate-600';

                    const textColor =
                        isHigh ? 'text-red-400'
                        : isMedium ? 'text-amber-400'
                        : 'text-ink-black-300';

                    return (
                        <div
                            key={item.name}
                            onClick={() => onSelectTarget(item.name)}
                            className={`flex flex-col gap-1.5 p-2.5 rounded-lg border transition-all cursor-pointer ${
                                isSelected ?
                                    'bg-ink-black-900 border-prussian-blue-500'
                                :   'bg-ink-black-900/60 border-ink-black-800 hover:border-ink-black-700'
                            }`}>
                            <div className='flex justify-between items-center text-xs font-sans'>
                                <span className='font-semibold text-ink-black-50'>
                                    {item.name}
                                </span>
                                <span
                                    className={`font-mono font-bold ${textColor}`}>
                                    {percentage}%
                                </span>
                            </div>
                            <div className='w-full bg-ink-black-800 rounded-full h-1.5 overflow-hidden'>
                                <div
                                    className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                                    style={{ width: `${percentage}%` }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
