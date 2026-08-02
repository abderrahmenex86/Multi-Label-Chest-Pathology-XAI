import React, { useState, type DragEvent, type ChangeEvent } from 'react';
import { Upload, AlertCircle, CheckCircle2 } from 'lucide-react';

interface FileDropzoneProps {
    onFileSelect: (file: File) => void;
    selectedFile: File | null;
}

export const FileDropzone: React.FC<FileDropzoneProps> = ({
    onFileSelect,
    selectedFile,
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
        <div className='flex-1 h-14 flex flex-col justify-center'>
            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative h-full border border-dashed rounded-xl px-3 py-2 text-center cursor-pointer transition-colors flex items-center justify-center gap-2.5 ${
                    isDragging ? 'border-prussian-blue-400 bg-ink-black-900'
                    : selectedFile ? 'border-prussian-blue-600 bg-ink-black-950'
                    : 'border-ink-black-800 bg-ink-black-950 hover:border-prussian-blue-500'
                }`}>
                <input
                    type='file'
                    accept='.png,.jpg,.jpeg'
                    onChange={handleFileChange}
                    className='absolute inset-0 w-full h-full opacity-0 cursor-pointer'
                />
                <div className='p-1.5 bg-ink-black-900 rounded-lg border border-ink-black-800 text-prussian-blue-400 shrink-0'>
                    {selectedFile ?
                        <CheckCircle2 className='w-4 h-4 text-prussian-blue-400' />
                    :   <Upload className='w-4 h-4' />}
                </div>
                <div className='flex flex-col text-left overflow-hidden'>
                    <p className='font-heading font-semibold text-ink-black-50 text-xs truncate'>
                        {selectedFile ?
                            selectedFile.name
                        :   'Upload Chest X-ray'}
                    </p>
                    <p className='font-sans text-xs text-ink-black-300 truncate'>
                        {selectedFile ?
                            `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                        :   'Drag & drop PNG or JPG (Max 5MB)'}
                    </p>
                </div>
            </div>

            {error && (
                <div className='mt-1 flex items-center gap-1 px-2 py-0.5 bg-red-950 border border-red-800 rounded text-red-200 text-xs font-sans'>
                    <AlertCircle className='w-3 h-3 shrink-0' />
                    <span className='truncate'>{error}</span>
                </div>
            )}
        </div>
    );
};
