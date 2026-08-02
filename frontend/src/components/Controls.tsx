import React, { type ChangeEvent } from 'react';
import { Sliders, Target, Layers } from 'lucide-react';
import type { Prediction } from '../types';

interface ControlsProps {
    opacity: number;
    setOpacity: (val: number) => void;
    target: string;
    predictions: Prediction[];
    onTargetChange: (val: string) => void;
    disabled: boolean;
}

export const Controls: React.FC<ControlsProps> = ({
    opacity,
    setOpacity,
    target,
    predictions,
    onTargetChange,
    disabled,
}) => {
    const handleTargetSelect = (e: ChangeEvent<HTMLSelectElement>) => {
        onTargetChange(e.target.value);
    };

    const eligible = predictions.filter(
        (item) => item.probability >= item.threshold
    );

    return (
        <div className='flex flex-col gap-3 p-4 bg-ink-black-950 rounded-xl border border-ink-black-800 shrink-0'>
            <div className='flex items-center justify-between pb-2 border-b border-ink-black-800'>
                <div className='flex items-center gap-2 text-prussian-blue-400'>
                    <Sliders className='w-4 h-4' />
                    <h3 className='font-heading font-semibold text-xs text-ink-black-50 uppercase tracking-wider'>
                        Visualization Controls
                    </h3>
                </div>
            </div>

            <div className='flex flex-col gap-1.5'>
                <div className='flex justify-between items-center text-xs font-sans'>
                    <label className='flex items-center gap-1 text-ink-black-200 font-medium'>
                        <Target className='w-3.5 h-3.5 text-prussian-blue-400' />
                        Target Pathology
                    </label>
                </div>
                <select
                    value={target}
                    onChange={handleTargetSelect}
                    disabled={disabled || eligible.length === 0}
                    className='w-full px-2.5 py-1.5 bg-ink-black-900 border border-ink-black-700 rounded-lg text-xs font-sans text-ink-black-50 focus:outline-none focus:border-prussian-blue-500 disabled:opacity-50 cursor-pointer'>
                    <option value=''>Auto-select top finding</option>
                    {eligible.map((item) => (
                        <option
                            key={item.name}
                            value={item.name}>
                            {item.name} ({(item.probability * 100).toFixed(1)}%)
                        </option>
                    ))}
                </select>
            </div>

            <div className='flex flex-col gap-1'>
                <div className='flex justify-between items-center text-xs font-sans'>
                    <label className='flex items-center gap-1 text-ink-black-200 font-medium'>
                        <Layers className='w-3 h-3 text-prussian-blue-400' />
                        Heatmap Opacity
                    </label>
                    <span className='text-ink-black-50 font-mono font-bold'>
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
                    className='w-full h-1.5 bg-ink-black-900 rounded-lg appearance-none cursor-pointer accent-prussian-blue-500 disabled:opacity-50'
                />
            </div>
        </div>
    );
};
