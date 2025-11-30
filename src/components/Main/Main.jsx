import React, { useState, useRef, useCallback, useEffect } from 'react'
import './Main.css'
import {assets} from '../../assets/assets.js'

// ============================================
// SAMPLE IMAGES - Just update these paths!
// Place your images in the public folder and update the paths below
// ============================================
const SAMPLE_IMAGES = [
  { name: 'Sample 1', src: '/sample1.png' },
  { name: 'Sample 2', src: '/sample2.png' },
  { name: 'Sample 3', src: '/sample3.png' },
  { name: 'Sample 4', src: '/sample4.png' },
  { name: 'Sample 5', src: '/sample5.png' },
  { name: 'Sample 6', src: '/sample6.png' },
  { name: 'Sample 7', src: '/sample7.png' },
  { name: 'Sample 8', src: '/sample8.png' },
]

const Main = () => {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [serverStatus, setServerStatus] = useState(null)
  const fileInputRef = useRef(null)

  // Check server health on mount
  useEffect(() => {
    checkServerHealth()
  }, [])

  const checkServerHealth = async () => {
    try {
      const response = await fetch('/api/health')
      if (response.ok) {
        const data = await response.json()
        setServerStatus(data)
        if (!data.model_loaded) {
          setError({
            title: 'Model Not Loaded',
            message: data.model_error || 'The segmentation model is not available.',
            suggestion: 'Please restart the backend server.'
          })
        }
      }
    } catch (err) {
      setServerStatus({ status: 'offline' })
      setError({
        title: 'Server Offline',
        message: 'Cannot connect to the backend server.',
        suggestion: 'Please make sure the backend is running: cd backend && python main.py'
      })
    }
  }

  // Handle file selection
  const handleFileSelect = (file) => {
    // Clear previous errors
    setError(null)
    
    if (!file) {
      setError({
        title: 'No File Selected',
        message: 'No file was provided.',
        suggestion: 'Please select an image file to upload.'
      })
      return
    }
    
    // Check file type
    if (!file.type.startsWith('image/')) {
      setError({
        title: 'Invalid File Type',
        message: `Selected file type "${file.type || 'unknown'}" is not supported.`,
        suggestion: 'Please upload an image file (JPEG, PNG, BMP, GIF, or WEBP).'
      })
      return
    }
    
    // Check file size (max 100MB)
    const maxSize = 100 * 1024 * 1024
    if (file.size > maxSize) {
      setError({
        title: 'File Too Large',
        message: `File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds the 100MB limit.`,
        suggestion: 'Please upload a smaller image or compress the file.'
      })
      return
    }
    
    // Check minimum size
    if (file.size < 100) {
      setError({
        title: 'File Too Small',
        message: 'The file appears to be corrupted or empty.',
        suggestion: 'Please try uploading a different image.'
      })
      return
    }
    
    setSelectedImage(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
  }

  // Handle file input change
  const handleInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }

  // Handle drag and drop
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFileSelect(file)
  }, [])

  // Handle paste from clipboard
  const handlePaste = useCallback((e) => {
    const items = e.clipboardData?.items
    if (items) {
      for (let item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) handleFileSelect(file)
          break
        }
      }
    }
  }, [])

  // Send image to API
  const handleSubmit = async () => {
    if (!selectedImage) {
      setError({
        title: 'No Image Selected',
        message: 'Please select an image before processing.',
        suggestion: 'Upload an image using drag & drop, paste, or the file browser.'
      })
      return
    }

    // Check server status
    if (serverStatus?.status === 'offline') {
      setError({
        title: 'Server Offline',
        message: 'The backend server is not running.',
        suggestion: 'Start the server with: cd backend && python main.py'
      })
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await fetch('/api/segment', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        // Handle structured error response
        setError({
          title: data.error || 'Processing Failed',
          message: data.message || 'An error occurred while processing the image.',
          suggestion: data.suggestion || 'Please try a different image.'
        })
        return
      }

      if (data.success) {
        setResult(data)
      } else {
        setError({
          title: 'Processing Failed',
          message: data.message || 'The server returned an unsuccessful response.',
          suggestion: data.suggestion || 'Please try again.'
        })
      }
    } catch (err) {
      console.error('Error:', err)
      
      // Handle network errors
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError({
          title: 'Connection Failed',
          message: 'Could not connect to the backend server.',
          suggestion: 'Make sure the backend is running: cd backend && python main.py'
        })
      } else if (err.name === 'SyntaxError') {
        setError({
          title: 'Invalid Response',
          message: 'The server returned an invalid response.',
          suggestion: 'Check the backend console for errors and try again.'
        })
      } else {
        setError({
          title: 'Unexpected Error',
          message: err.message || 'An unexpected error occurred.',
          suggestion: 'Please try again or contact support.'
        })
      }
    } finally {
      setLoading(false)
    }
  }

  // Open file picker
  const openFilePicker = () => {
    fileInputRef.current?.click()
  }

  // Reset state
  const handleReset = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }

  // Dismiss error
  const dismissError = () => {
    setError(null)
  }

  // Load sample image
  const loadSampleImage = async (sampleSrc) => {
    setError(null)
    try {
      const response = await fetch(sampleSrc)
      if (!response.ok) {
        throw new Error('Failed to fetch sample image')
      }
      const blob = await response.blob()
      const file = new File([blob], 'sample.png', { type: 'image/png' })
      handleFileSelect(file)
    } catch (err) {
      setError({
        title: 'Failed to Load Sample',
        message: 'Could not load the sample image.',
        suggestion: 'Please try uploading your own image instead.'
      })
    }
  }

  // Retry with server health check
  const handleRetry = async () => {
    setError(null)
    await checkServerHealth()
  }

  // Handle drag start for sample images
  const handleSampleDragStart = (e, sampleSrc) => {
    e.dataTransfer.setData('sampleImage', sampleSrc)
  }

  // Handle drop from sample panel
  const handleDropWithSample = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const sampleSrc = e.dataTransfer.getData('sampleImage')
    if (sampleSrc) {
      loadSampleImage(sampleSrc)
      return
    }
    
    const file = e.dataTransfer.files?.[0]
    if (file) handleFileSelect(file)
  }, [])

  return (
    <div className='main' onPaste={handlePaste}>
        <div className="nav">
            <p>RunwAI</p>
            <div className="nav-right">
              {serverStatus && (
                <span className={`server-status ${serverStatus.model_loaded ? 'online' : 'offline'}`}>
                  {serverStatus.model_loaded ? '● Online' : '● Offline'}
                </span>
              )}
              <img src={assets.user_icon} alt="user_icon"/>
            </div>
        </div>
        
        <div className="main-container">
            {!result ? (
              <>
                <div className="greet">
                    <p><span>Hello, Welcome to RunwAI.</span></p>
                    <p className="subtitle">A high-precision semantic segmentation model for runway detection in aerial imagery, trained on 1920x1080 resolution images.</p>
                </div>

                {/* Dataset & Compatibility Info */}
                <div className="info-banners">
                  <div className="dataset-info">
                    <div className="dataset-badge">
                      <span className="dataset-icon">🎮</span>
                      <span className="dataset-name">FS2020 Dataset</span>
                    </div>
                    <p className="dataset-description">
                      Model trained on <a href="https://www.kaggle.com/datasets/relufrank/fs2020-runway-dataset" target="_blank" rel="noopener noreferrer">Kaggle FS2020 Runway Segmentation</a> synthetic aerial images.
                    </p>
                  </div>

                  <div className="compatibility-notice">
                    <div className="notice-header">
                      <span className="notice-icon">⚠️</span>
                      <span className="notice-title">Compatibility Note</span>
                    </div>
                    <p className="notice-text">
                      This model was trained on <strong>Flight Simulator 2020</strong> synthetic imagery. 
                      It works best with FS2020 screenshots and may have reduced accuracy on real-world aerial photos.
                    </p>
                  </div>
                </div>
                
                {/* Error Display */}
                {error && (
                  <div className="error-container">
                    <div className="error-content">
                      <div className="error-icon">⚠️</div>
                      <div className="error-text">
                        <h4 className="error-title">{error.title}</h4>
                        <p className="error-message">{error.message}</p>
                        {error.suggestion && (
                          <p className="error-suggestion">💡 {error.suggestion}</p>
                        )}
                      </div>
                      <button className="error-dismiss" onClick={dismissError}>×</button>
                    </div>
                    {error.title.includes('Server') && (
                      <button className="retry-btn" onClick={handleRetry}>
                        🔄 Retry Connection
                      </button>
                    )}
                  </div>
                )}
                
                {/* Upload Area */}
                <div 
                  className={`upload-area ${isDragOver ? 'drag-over' : ''} ${previewUrl ? 'has-image' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDropWithSample}
                  onClick={!previewUrl ? openFilePicker : undefined}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef}
                    onChange={handleInputChange}
                    accept="image/*"
                    style={{ display: 'none' }}
                  />
                  
                  {previewUrl ? (
                    <div className="preview-container">
                      <img src={previewUrl} alt="Preview" className="preview-image" />
                      <button className="change-image-btn" onClick={(e) => { e.stopPropagation(); openFilePicker(); }}>
                        Change Image
                      </button>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <img src={assets.gallery_icon} alt="upload" className="upload-icon" />
                      <p>Drag and drop aerial imagery, paste from clipboard,<br/>or browse files to upload.</p>
                      <span className="supported-formats">Supports: JPEG, PNG, BMP, GIF, WEBP (max 100MB)</span>
                    </div>
                  )}
                  
                  {previewUrl && (
                    <button 
                      className="send-btn" 
                      onClick={(e) => { e.stopPropagation(); handleSubmit(); }}
                      disabled={loading || serverStatus?.status === 'offline'}
                      title={serverStatus?.status === 'offline' ? 'Server is offline' : 'Process image'}
                    >
                      {loading ? (
                        <div className="spinner"></div>
                      ) : (
                        <img src={assets.send_icon} alt="send" />
                      )}
                    </button>
                  )}
                </div>

                {/* Loading State */}
                {loading && (
                  <div className="loading-info">
                    <p>🔍 Analyzing image for runway segmentation...</p>
                    <p className="loading-subtext">This may take a few seconds depending on image size.</p>
                  </div>
                )}

                {/* Sample Images */}
                <div className="cards">
                    {SAMPLE_IMAGES.map((sample, index) => (
                      <div 
                        key={index}
                        className="card sample-card" 
                        onClick={() => loadSampleImage(sample.src)}
                      >
                          <img src={sample.src} alt={sample.name} className="sample-thumbnail"/>
                          <p>{sample.name}</p>
                      </div>
                    ))}
                </div>
              </>
            ) : (
              /* Results View */
              <div className="results-container">
                <div className="results-header">
                  <h2>Segmentation Results</h2>
                  <button className="back-btn" onClick={handleReset}>
                    ← Upload New Image
                  </button>
                </div>
                
                <div className="results-grid">
                  <div className="result-card">
                    <h3>Original Image</h3>
                    <img src={result.original} alt="Original" />
                  </div>
                  <div className="result-card">
                    <h3>Segmented Overlay</h3>
                    <img src={result.overlay} alt="Overlay" />
                  </div>
                  <div className="result-card">
                    <h3>Binary Mask</h3>
                    <img src={result.mask} alt="Mask" />
                  </div>
                  <div className="result-card stats-card">
                    <h3>Statistics</h3>
                    <div className="stats">
                      <div className="stat-item">
                        <span className="stat-label">Runway Pixels</span>
                        <span className="stat-value">{result.stats.runway_pixels.toLocaleString()}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Total Pixels</span>
                        <span className="stat-value">{result.stats.total_pixels.toLocaleString()}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Runway Coverage</span>
                        <span className="stat-value highlight">{result.stats.runway_percentage}%</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Image Size</span>
                        <span className="stat-value">{result.image_size.width} × {result.image_size.height}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
        </div>
    </div>
  )
}

export default Main