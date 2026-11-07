import React, { useState, type ChangeEvent, type DragEvent } from 'react';
import { AlertCircle, CheckCircle2, RefreshCw, Upload } from 'lucide-react';

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
            setError('Choose a PNG or JPEG image.');
            return;
        }
        if (file.size > 5242880) {
            setError('That file is larger than 5 MB.');
            return;
        }
        onFileSelect(file);
    };

    const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLLabelElement>) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
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
        <div className='shrink-0 rounded-2xl border border-line bg-surface p-4'>
            <div className='flex items-center justify-between gap-3'>
                <h2 className='font-heading text-sm font-semibold text-ink'>
                    Image
                </h2>
                {selectedFile && (
                    <span className='text-[11px] font-medium tabular-nums text-muted'>
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                )}
            </div>

            <label
                htmlFor='xray-file-input'
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`mt-3 flex cursor-pointer items-center gap-3 rounded-xl border border-dashed bg-white px-3 py-3 transition-colors ${
                    isDragging
                        ? 'border-brand bg-brand-soft'
                        : selectedFile
                          ? 'border-brand/40'
                          : 'border-line hover:border-brand/50 hover:bg-brand-soft/40'
                }`}>
                <span
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                        selectedFile
                            ? 'bg-brand-soft text-brand'
                            : 'bg-surface text-muted'
                    }`}>
                    {selectedFile ?
                        <CheckCircle2 className='h-4 w-4' strokeWidth={2} />
                    :   <Upload className='h-4 w-4' strokeWidth={1.8} />}
                </span>

                <span className='min-w-0 flex-1'>
                    <span className='block truncate text-sm font-medium text-ink'>
                        {selectedFile ?
                            selectedFile.name
                        :   'Select a chest X-ray'}
                    </span>
                    <span className='mt-0.5 block text-xs text-muted'>
                        {selectedFile ?
                            'Ready to analyze'
                        :   'PNG or JPG, up to 5 MB'}
                    </span>
                </span>

                <span className='shrink-0 text-xs font-medium text-brand'>
                    {selectedFile ? 'Change' : 'Browse'}
                </span>

                <input
                    id='xray-file-input'
                    type='file'
                    accept='.png,.jpg,.jpeg'
                    onChange={handleFileChange}
                    className='sr-only'
                />
            </label>

            {error && (
                <div
                    role='alert'
                    className='mt-2 flex items-center gap-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700'>
                    <AlertCircle className='h-3.5 w-3.5 shrink-0' />
                    <span>{error}</span>
                </div>
            )}

            <button
                type='button'
                onClick={onSubmit}
                disabled={loading || !selectedFile}
                className='mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-brand-dark disabled:cursor-not-allowed disabled:bg-line disabled:text-muted'>
                {loading ?
                    <>
                        <RefreshCw className='h-4 w-4 animate-spin' />
                        <span>Analyzing...</span>
                    </>
                :   <span>Analyze X-ray</span>}
            </button>
        </div>
    );
};
