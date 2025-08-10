import React, { useState, useEffect } from 'react';
import { AppLogger } from '@/lib/logger';
import { callPythonMethod, uploadFile, onPythonEvent, offPythonEvent } from '@/lib/pythonBridge';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';

interface ProcessedDocument {
  id: string;
  name: string;
  status: 'uploading' | 'uploaded' | 'processing' | 'processed' | 'failed';
  size: number;
  type: string;
  uploadTime: Date;
  processTime?: Date;
  concepts?: string[];
  complexity?: number;
  readability?: number;
  error?: string;
}

interface DeepSeekDocumentProcessorProps {
  onDocumentProcessed?: (docId: string, docName: string, concepts?: string[]) => void;
  maxFileSize?: number;
  allowedTypes?: string[];
}

const DeepSeekDocumentProcessor: React.FC<DeepSeekDocumentProcessorProps> = ({ 
  onDocumentProcessed,
  maxFileSize = 10 * 1024 * 1024, // 10MB default
  allowedTypes = ['.pdf', '.txt', '.docx', '.md']
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [processedDocuments, setProcessedDocuments] = useState<ProcessedDocument[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<{ [key: string]: number }>({});

  // Listen for processing progress events
  useEffect(() => {
    const handleProcessingProgress = (data: {
      document_id: string;
      stage: string;
      progress: number;
      details?: string;
    }) => {
      setProcessingProgress(prev => ({
        ...prev,
        [data.document_id]: data.progress
      }));

      // Update document status
      setProcessedDocuments(prev => prev.map(doc => {
        if (doc.id === data.document_id) {
          return {
            ...doc,
            status: data.progress >= 100 ? 'processed' : 'processing'
          };
        }
        return doc;
      }));

      AppLogger.debug('DEEPSEEK_DOC_PROCESSOR', 'Processing progress update', data);
    };

    const handleProcessingComplete = (data: {
      document_id: string;
      concepts: string[];
      complexity: number;
      readability: number;
    }) => {
      setProcessedDocuments(prev => prev.map(doc => {
        if (doc.id === data.document_id) {
          const updatedDoc = {
            ...doc,
            status: 'processed' as const,
            processTime: new Date(),
            concepts: data.concepts,
            complexity: data.complexity,
            readability: data.readability
          };

          if (onDocumentProcessed) {
            onDocumentProcessed(doc.id, doc.name, data.concepts);
          }

          return updatedDoc;
        }
        return doc;
      }));

      // Remove from processing progress
      setProcessingProgress(prev => {
        const updated = { ...prev };
        delete updated[data.document_id];
        return updated;
      });

      AppLogger.success('DEEPSEEK_DOC_PROCESSOR', 'Document processing completed', data);
    };

    onPythonEvent('deepseek_document_progress', handleProcessingProgress);
    onPythonEvent('deepseek_document_complete', handleProcessingComplete);

    return () => {
      offPythonEvent('deepseek_document_progress', handleProcessingProgress);
      offPythonEvent('deepseek_document_complete', handleProcessingComplete);
    };
  }, [onDocumentProcessed]);

  // Validate file
  const validateFile = (file: File): string | null => {
    if (file.size > maxFileSize) {
      return `File size exceeds ${(maxFileSize / (1024 * 1024)).toFixed(1)}MB limit`;
    }

    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(fileExtension)) {
      return `File type ${fileExtension} not supported. Allowed types: ${allowedTypes.join(', ')}`;
    }

    return null;
  };

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    handleFiles(files);
  };

  // Handle drag and drop
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  // Process selected files
  const handleFiles = (files: File[]) => {
    const validFiles: File[] = [];
    const errors: string[] = [];

    files.forEach(file => {
      const error = validateFile(file);
      if (error) {
        errors.push(`${file.name}: ${error}`);
      } else {
        validFiles.push(file);
      }
    });

    if (errors.length > 0) {
      setError(errors.join('; '));
    } else {
      setError(null);
    }

    setSelectedFiles(prev => [...prev, ...validFiles]);
    
    AppLogger.info('DEEPSEEK_DOC_PROCESSOR', 'Files selected for processing', {
      validFiles: validFiles.length,
      errors: errors.length
    });
  };

  // Upload and process all selected files
  const handleUploadAll = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select files first.');
      return;
    }

    setError(null);

    for (const file of selectedFiles) {
      await uploadAndProcessFile(file);
    }

    // Clear selected files after processing
    setSelectedFiles([]);
  };

  // Upload and process a single file
  const uploadAndProcessFile = async (file: File) => {
    const docId = `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Add to processed documents with uploading status
    const newDoc: ProcessedDocument = {
      id: docId,
      name: file.name,
      status: 'uploading',
      size: file.size,
      type: file.type,
      uploadTime: new Date()
    };

    setProcessedDocuments(prev => [...prev, newDoc]);

    try {
      AppLogger.info('DEEPSEEK_DOC_PROCESSOR', 'Starting file upload and processing', {
        fileName: file.name,
        fileSize: file.size,
        docId
      });

      // Upload file
      const uploadResponse = await uploadFile(file);
      
      // Update status to uploaded
      setProcessedDocuments(prev => prev.map(doc => 
        doc.id === docId ? { ...doc, status: 'uploaded' } : doc
      ));

      // Start processing
      const actualDocId = uploadResponse.document_id || uploadResponse.file_path || docId;
      await handleProcessDocument(actualDocId, file.name, docId);

    } catch (err) {
      const errorMessage = (err as Error).message;
      setProcessedDocuments(prev => prev.map(doc => 
        doc.id === docId ? { ...doc, status: 'failed', error: errorMessage } : doc
      ));
      
      AppLogger.error('DEEPSEEK_DOC_PROCESSOR', 'File upload/processing failed', {
        fileName: file.name,
        error: errorMessage
      });
    }
  };

  const handleProcessDocument = async (actualDocId: string, docName: string, localDocId: string) => {
    // Update status to processing
    setProcessedDocuments(prev => prev.map(doc => 
      doc.id === localDocId ? { ...doc, status: 'processing', id: actualDocId } : doc
    ));

    try {
      AppLogger.info('DEEPSEEK_DOC_PROCESSOR', 'Starting document processing', {
        actualDocId,
        docName
      });

      // Call Python backend to process the document with DeepSeek
      const response = await callPythonMethod('process_document_for_deepseek', {
        document_id: actualDocId,
        enable_concept_extraction: true,
        enable_complexity_analysis: true,
        enable_readability_analysis: true
      });

      AppLogger.success('DEEPSEEK_DOC_PROCESSOR', 'Document processing initiated', response);

      // The actual completion will be handled by the event listener
      // This just initiates the processing

    } catch (err) {
      const errorMessage = (err as Error).message;
      setProcessedDocuments(prev => prev.map(doc => 
        doc.id === actualDocId ? { ...doc, status: 'failed', error: errorMessage } : doc
      ));
      
      AppLogger.error('DEEPSEEK_DOC_PROCESSOR', 'Document processing failed', {
        actualDocId,
        error: errorMessage
      });
    }
  };

  // Remove a document
  const removeDocument = (docId: string) => {
    setProcessedDocuments(prev => prev.filter(doc => doc.id !== docId));
    setProcessingProgress(prev => {
      const updated = { ...prev };
      delete updated[docId];
      return updated;
    });
    
    AppLogger.info('DEEPSEEK_DOC_PROCESSOR', 'Document removed', { docId });
  };

  // Remove a selected file
  const removeSelectedFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Get status color
  const getStatusColor = (status: ProcessedDocument['status']) => {
    switch (status) {
      case 'processed': return 'text-green-400 bg-green-500/20';
      case 'processing': return 'text-blue-400 bg-blue-500/20';
      case 'uploaded': return 'text-yellow-400 bg-yellow-500/20';
      case 'uploading': return 'text-purple-400 bg-purple-500/20';
      case 'failed': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  // Get status icon
  const getStatusIcon = (status: ProcessedDocument['status']) => {
    switch (status) {
      case 'processed': return '✅';
      case 'processing': return '🔄';
      case 'uploaded': return '📤';
      case 'uploading': return '⬆️';
      case 'failed': return '❌';
      default: return '⏳';
    }
  };

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="deepseek-document-processor bg-gray-800/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="text-2xl">📄</div>
          <div>
            <h3 className="text-xl font-semibold text-white">Document Processing</h3>
            <p className="text-sm text-gray-400">Upload documents for DeepSeek analysis</p>
          </div>
        </div>
        <div className="text-sm text-gray-400">
          Max: {(maxFileSize / (1024 * 1024)).toFixed(1)}MB | Types: {allowedTypes.join(', ')}
        </div>
      </div>

      {/* Drag & Drop Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-purple-400 bg-purple-500/10' 
            : 'border-gray-600 hover:border-gray-500'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="text-4xl mb-3">📁</div>
        <p className="text-white font-medium mb-2">
          {dragActive ? 'Drop files here' : 'Drag & drop files here'}
        </p>
        <p className="text-gray-400 text-sm mb-4">or</p>
        <input
          type="file"
          multiple
          accept={allowedTypes.join(',')}
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="inline-flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 cursor-pointer transition-colors"
        >
          <span className="mr-2">📎</span>
          Choose Files
        </label>
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-semibold text-white">Selected Files ({selectedFiles.length})</h4>
            <button
              onClick={handleUploadAll}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Upload & Process All
            </button>
          </div>
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-700/50 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <div className="text-lg">📄</div>
                  <div>
                    <div className="text-white font-medium">{file.name}</div>
                    <div className="text-sm text-gray-400">{formatFileSize(file.size)}</div>
                  </div>
                </div>
                <button
                  onClick={() => removeSelectedFile(index)}
                  className="text-red-400 hover:text-red-300 transition-colors"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 bg-red-500/20 border border-red-500/50 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-300">
            <span className="text-lg">⚠️</span>
            <span className="font-medium">Error:</span>
          </div>
          <p className="text-red-200 mt-1">{error}</p>
        </div>
      )}

      {/* Processed Documents */}
      {processedDocuments.length > 0 && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold text-white mb-4">Processed Documents</h4>
          <div className="space-y-3">
            {processedDocuments.map((doc) => (
              <div key={doc.id} className="bg-gray-700/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className="text-lg">{getStatusIcon(doc.status)}</span>
                    <div>
                      <div className="text-white font-medium">{doc.name}</div>
                      <div className="text-sm text-gray-400">
                        {formatFileSize(doc.size)} • Uploaded {doc.uploadTime.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(doc.status)}`}>
                      {doc.status.toUpperCase()}
                    </span>
                    <button
                      onClick={() => removeDocument(doc.id)}
                      className="text-gray-400 hover:text-red-400 transition-colors"
                    >
                      🗑️
                    </button>
                  </div>
                </div>

                {/* Processing Progress */}
                {doc.status === 'processing' && processingProgress[doc.id] !== undefined && (
                  <div className="mb-3">
                    <div className="flex justify-between text-sm text-gray-300 mb-1">
                      <span>Processing...</span>
                      <span>{processingProgress[doc.id].toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${processingProgress[doc.id]}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {/* Document Analysis Results */}
                {doc.status === 'processed' && (
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-4">
                    {doc.complexity !== undefined && (
                      <div className="text-center">
                        <div className="text-lg font-bold text-purple-300">{doc.complexity}/10</div>
                        <div className="text-xs text-gray-400">Complexity</div>
                      </div>
                    )}
                    {doc.readability !== undefined && (
                      <div className="text-center">
                        <div className="text-lg font-bold text-blue-300">{doc.readability}%</div>
                        <div className="text-xs text-gray-400">Readability</div>
                      </div>
                    )}
                    {doc.concepts && (
                      <div className="text-center">
                        <div className="text-lg font-bold text-green-300">{doc.concepts.length}</div>
                        <div className="text-xs text-gray-400">Concepts</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Extracted Concepts */}
                {doc.concepts && doc.concepts.length > 0 && (
                  <div className="mt-3">
                    <div className="text-sm font-medium text-gray-300 mb-2">Key Concepts:</div>
                    <div className="flex flex-wrap gap-2">
                      {doc.concepts.slice(0, 8).map((concept, index) => (
                        <span 
                          key={index}
                          className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs"
                        >
                          {concept}
                        </span>
                      ))}
                      {doc.concepts.length > 8 && (
                        <span className="px-2 py-1 bg-gray-500/20 text-gray-400 rounded-full text-xs">
                          +{doc.concepts.length - 8} more
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Error Details */}
                {doc.status === 'failed' && doc.error && (
                  <div className="mt-3 text-sm text-red-300 bg-red-500/10 rounded p-2">
                    <strong>Error:</strong> {doc.error}
                  </div>
                )}

                {/* Processing Time */}
                {doc.processTime && (
                  <div className="mt-2 text-xs text-gray-400">
                    Processed in {((doc.processTime.getTime() - doc.uploadTime.getTime()) / 1000).toFixed(1)}s
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Usage Tips */}
      {processedDocuments.length === 0 && selectedFiles.length === 0 && (
        <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2 text-blue-300 mb-2">
            <span className="text-lg">💡</span>
            <span className="font-medium">Tips for better results:</span>
          </div>
          <ul className="text-blue-200 text-sm space-y-1">
            <li>• Upload academic papers, textbooks, or research documents</li>
            <li>• Ensure documents are text-based (not scanned images)</li>
            <li>• Higher complexity documents generate better expert questions</li>
            <li>• Multiple documents can be processed simultaneously</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default DeepSeekDocumentProcessor;
