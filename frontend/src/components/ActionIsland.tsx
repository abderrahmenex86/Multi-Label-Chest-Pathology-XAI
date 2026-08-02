import React, { useState, type ChangeEvent, type DragEvent } from 'react';
import {
    Upload,
    Play,
    RefreshCw,
    CheckCircle2,
    AlertCircle,
} from 'lucide-react';

interface ActionIslandProps {
    onFileSelect: (file: File) => void;
    selectedFile: File | null;
    onSubmit: () => void;
    loading: boolean;
}

export const ActionIsland: React.FC<ActionIslandProps> = ({
    onFileSelect,
    selectedFile,
    onSubmit,
    loading,
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const validateAndPass = (file: File) => {
        setError(null);
        if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
            setError('Only PNG, JPG, and JPEG allowed.');
            return;
        }
        if (file.size > 5242880) {
            setError('File size exceeds 5MB limit.');
            return;
        }
        onFileSelect(file);
    };

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            validateAndPass(e.dataTransfer.files[0]);
        }
    };

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            validateAndPass(e.target.files[0]);
        }
    };

    return (
        <div className='bg-slate-dark border border-steel-blue/40 rounded-xl p-4 flex flex-col gap-3 shrink-0'>
            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border border-dashed rounded-lg p-3 text-center cursor-pointer transition-colors flex items-center justify-center gap-2 ${
                    isDragging ? 'border-coral-orange bg-steel-blue/20'
                    : selectedFile ? 'border-coral-orange/60 bg-steel-blue/10'
                    : 'border-silver-gray/40 hover:border-coral-orange/60 bg-steel-blue/10'
                }`}>
                <input
                    type='file'
                    accept='.png,.jpg,.jpeg'
                    onChange={handleFileChange}
                    className='absolute inset-0 w-full h-full opacity-0 cursor-pointer'
                />
                <div className='p-1.5 rounded bg-steel-blue/30 text-coral-orange shrink-0'>
                    {selectedFile ?
                        <CheckCircle2 className='w-4 h-4 text-coral-orange' />
                    :   <Upload className='w-4 h-4' />}
                </div>
                <div className='flex flex-col text-left overflow-hidden'>
                    <span className='font-heading font-semibold text-pure-white text-xs truncate'>
                        {selectedFile ?
                            selectedFile.name
                        :   'Select or Drop X-ray'}
                    </span>
                    <span className='font-sans text-xs text-silver-gray truncate'>
                        {selectedFile ?
                            `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                        :   'Max 5MB'}
                    </span>
                </div>
            </div>

            {error && (
                <div className='flex items-center gap-1.5 px-2 py-1 bg-red-950/60 border border-red-800/60 rounded text-red-200 text-xs font-sans'>
                    <AlertCircle className='w-3.5 h-3.5 shrink-0' />
                    <span className='truncate'>{error}</span>
                </div>
            )}

            <div className='grid grid-cols-2 gap-2'>
                <label className='flex items-center justify-center gap-1.5 px-3 py-2 bg-steel-blue/30 hover:bg-steel-blue/50 text-pure-white font-heading font-semibold text-xs rounded-lg border border-steel-blue/50 cursor-pointer transition-colors'>
                    <Upload className='w-3.5 h-3.5 text-silver-gray' />
                    <span>Upload</span>
                    <input
                        type='file'
                        accept='.png,.jpg,.jpeg'
                        onChange={handleFileChange}
                        className='hidden'
                    />
                </label>

                <button
                    onClick={onSubmit}
                    disabled={loading || !selectedFile}
                    className='flex items-center justify-center gap-1.5 px-3 py-2 bg-coral-orange hover:bg-coral-orange/90 disabled:bg-steel-blue/20 disabled:text-silver-gray/40 disabled:border-transparent text-pure-white font-heading font-semibold text-xs rounded-lg transition-colors border border-coral-orange/80 cursor-pointer'>
                    {loading ?
                        <>
                            <RefreshCw className='w-3.5 h-3.5 animate-spin' />
                            <span>Running...</span>
                        </>
                    :   <>
                            <Play className='w-3.5 h-3.5 fill-current' />
                            <span>Submit</span>
                        </>
                    }
                </button>
            </div>
        </div>
    );
};
