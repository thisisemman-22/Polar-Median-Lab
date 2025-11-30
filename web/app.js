const form = document.getElementById('denoise-form');
const kernelValue = document.getElementById('kernel-value');
const noiseValue = document.getElementById('noise-value');
const statusEl = document.getElementById('status');
const originalPreview = document.getElementById('preview-original');
const noisyPreview = document.getElementById('preview-noisy');
const denoisedPreview = document.getElementById('preview-denoised');
const runtimeEl = document.getElementById('metric-runtime');
const psnrEl = document.getElementById('metric-psnr');
const modeEl = document.getElementById('metric-mode');

const fileInput = form.elements.namedItem('image');
const kernelInput = form.elements.namedItem('kernel');
const noiseInput = form.elements.namedItem('noise');
const addNoiseInput = document.getElementById('add-noise');
const submitButton = form.querySelector('.cta');

const formControls = Array.from(form.elements);

kernelInput.addEventListener('input', () => {
  kernelValue.textContent = kernelInput.value;
});

noiseInput.addEventListener('input', () => {
  noiseValue.textContent = `${noiseInput.value}%`;
  addNoiseInput.checked = noiseInput.value !== '0';
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    statusEl.textContent = 'Please choose an image first.';
    return;
  }
  statusEl.textContent = 'Processing image with festive magic…';
  const formData = new FormData(form);
  formData.set('kernel', ensureOddKernel(formData.get('kernel')));
  formData.set('noise', String(Number(formData.get('noise')) / 100));
  formData.set('add_noise', addNoiseInput.checked ? 'true' : 'false');
  setProcessingState(true);

  try {
    const response = await fetch('/api/denoise', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || 'Server error');
    }
    const payload = await response.json();
    setPreviewImage(originalPreview, payload.original);
    setPreviewImage(noisyPreview, payload.noisy);
    setPreviewImage(denoisedPreview, payload.processed);
    runtimeEl.textContent = formatMetric(payload.runtime_ms, {
      suffix: ' ms',
      digits: 1,
      fallback: '—',
    });
    psnrEl.textContent = formatPsnr(payload.psnr);
    modeEl.textContent = payload.strategy === 'optimized' ? 'Dual Heap' : 'Brute Force';
    statusEl.textContent = 'Done! Export your denoised image below.';
  } catch (error) {
    console.error(error);
    statusEl.textContent = `Something went wrong: ${error.message}`;
  } finally {
    setProcessingState(false);
  }
});

function ensureOddKernel(value) {
  const n = Number(value);
  return n % 2 === 0 ? n + 1 : n;
}

function formatMetric(value, { digits = 1, suffix = '', fallback = '—' } = {}) {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : fallback;
}

function formatPsnr(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} dB` : 'Perfect (∞ dB)';
}

function setPreviewImage(element, dataUrl) {
  if (!dataUrl) {
    element.removeAttribute('src');
    element.classList.remove('preview-loaded');
    return;
  }
  element.src = dataUrl;
  element.classList.add('preview-loaded');
}

function setProcessingState(isProcessing) {
  if (submitButton) {
    submitButton.disabled = isProcessing;
    submitButton.classList.toggle('loading', isProcessing);
    submitButton.textContent = isProcessing ? 'Working…' : 'Denoise Image';
  }
  formControls.forEach((control) => {
    if (control !== submitButton) {
      control.disabled = isProcessing;
    }
  });
}
