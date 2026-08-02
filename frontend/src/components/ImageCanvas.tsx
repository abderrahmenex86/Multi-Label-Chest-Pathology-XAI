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
        <div className='relative flex-1 min-h-0 w-full bg-slate-dark rounded-xl overflow-hidden border border-steel-blue/40 flex items-center justify-center p-2'>
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

                    <div className='absolute bottom-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-slate-dark/90 border border-steel-blue/50 rounded-lg text-silver-gray text-xs font-sans backdrop-blur-sm shadow-md'>
                        <Eye className='w-3.5 h-3.5 text-coral-orange' />
                        <span>
                            {heatmap ? 'Heatmap Overlay Active' : 'Base View'}
                        </span>
                    </div>
                </>
            :   <div className='flex flex-col items-center justify-center gap-3 text-silver-gray/60 font-sans'>
                    <div className='p-4 bg-steel-blue/20 rounded-full border border-steel-blue/30 text-silver-gray'>
                        <FileSearch className='w-10 h-10' />
                    </div>
                    <p className='text-sm font-medium text-pure-white'>
                        No Image Loaded
                    </p>
                    <p className='text-xs text-silver-gray'>
                        Upload a chest X-ray image from the action island on the
                        right
                    </p>
                </div>
            }
        </div>
    );
};
