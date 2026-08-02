import React from 'react';
import { Eye, FileSearch } from 'lucide-react';

interface ImageCanvasProps {
    preview: string | null;
    heatmap: string | null;
    opacity: number;
}

export const ImageCanvas: React.FC<ImageCanvasProps> = ({
    preview,
    heatmap,
    opacity,
}) => {
    return (
        <div className='relative flex-1 min-h-0 w-full bg-ink-black-950 rounded-xl overflow-hidden border border-ink-black-800 flex items-center justify-center p-2'>
            {preview ?
                <>
                    <img
                        src={preview}
                        alt='Chest X-ray Base'
                        className='w-full h-full object-contain block rounded-lg'
                    />

                    {heatmap && (
                        <img
                            src={heatmap}
                            alt='HiResCAM Heatmap'
                            className='absolute inset-0 w-full h-full object-contain pointer-events-none p-2 transition-opacity duration-75'
                            style={{ opacity }}
                        />
                    )}

                    <div className='absolute bottom-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-ink-black-900 border border-ink-black-800 rounded-lg text-ink-black-300 text-xs font-sans opacity-90 shadow-md'>
                        <Eye className='w-3.5 h-3.5 text-prussian-blue-400' />
                        <span>
                            {heatmap ? 'Heatmap Overlay Active' : 'Base View'}
                        </span>
                    </div>
                </>
            :   <div className='flex flex-col items-center justify-center gap-3 text-ink-black-400 font-sans'>
                    <div className='p-4 bg-ink-black-900 rounded-full border border-ink-black-800 text-ink-black-700'>
                        <FileSearch className='w-10 h-10' />
                    </div>
                    <p className='text-sm font-medium text-ink-black-300'>
                        No Image Uploaded
                    </p>
                    <p className='text-xs text-ink-black-300'>
                        Upload a chest X-ray image below to begin analysis
                    </p>
                </div>
            }
        </div>
    );
};
