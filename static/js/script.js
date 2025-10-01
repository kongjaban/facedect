document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  if (path.includes('/login')) {
    handleAuthPage();
  } else if (path === '/') {
    handleMainPage();
  }
});

function handleAuthPage() {
  const loginContainer = document.getElementById('login-container');
  const signupContainer = document.getElementById('signup-container');
  const showSignup = document.getElementById('show-signup');
  const showLogin = document.getElementById('show-login');
  const loginForm = document.getElementById('login-form');
  const signupForm = document.getElementById('signup-form');
  const errorMessage = document.getElementById('error-message');

  showSignup.addEventListener('click', (e) => {
    e.preventDefault();
    loginContainer.classList.add('hidden');
    signupContainer.classList.remove('hidden');
    errorMessage.textContent = '';
  });

  showLogin.addEventListener('click', (e) => {
    e.preventDefault();
    signupContainer.classList.add('hidden');
    loginContainer.classList.remove('hidden');
    errorMessage.textContent = '';
  });

  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(loginForm);
    const data = Object.fromEntries(formData.entries());
    const res = await fetch('/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    const result = await res.json();
    if (result.status === 'success') window.location.href = '/';
    else errorMessage.textContent = result.message || '로그인 실패';
  });

  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(signupForm);
    const data = Object.fromEntries(formData.entries());
    const res = await fetch('/signup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    const result = await res.json();
    if (result.status === 'success') window.location.href = '/';
    else errorMessage.textContent = result.message || '회원가입 실패';
  });
}

function handleMainPage() {
  const userNameSpan = document.getElementById('user-name');
  const uploadCard = document.getElementById('upload-card');
  const loadingCard = document.getElementById('loading-card');
  const resultCard = document.getElementById('result-card');
  const uploadForm = document.getElementById('upload-form');
  const imageInput = document.getElementById('image-input');
  const fileNameSpan = document.getElementById('file-name');
  const resetButton = document.getElementById('reset-button');
  const errorMessage = document.getElementById('error-message');

  fetch('/api/user').then(r => r.json()).then(d => {
    if (d.status === 'success' && d.user) userNameSpan.textContent = `${d.user.name || d.user.email}님, 안녕하세요.`;
  }).catch(() => {});

  imageInput.addEventListener('change', () => {
    fileNameSpan.textContent = imageInput.files.length > 0 ? imageInput.files[0].name : '선택된 파일 없음';
  });

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!imageInput.files[0]) {
      errorMessage.textContent = '분석할 이미지 파일을 선택해주세요.';
      return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    // Append optional fields
    const genderEl = document.getElementById('gender');
    const birthDateEl = document.getElementById('birth_date');
    const birthTimeEl = document.getElementById('birth_time');
    if (genderEl && genderEl.value) formData.append('gender', genderEl.value);
    if (birthDateEl && birthDateEl.value) formData.append('birth_date', birthDateEl.value);
    if (birthTimeEl && birthTimeEl.value) formData.append('birth_time', birthTimeEl.value);

    uploadCard.classList.add('hidden');
    resultCard.classList.add('hidden');
    loadingCard.classList.remove('hidden');
    // Progress bar setup
    const progressEl = document.getElementById('loading-progress');
    const loadingTextEl = document.getElementById('loading-text');
    if (progressEl) progressEl.style.width = '0%';
    let elapsedMs = 0;
    const totalMs = 60000; // 60 seconds
    const tickMs = 200; // update every 200ms
    if (loadingTextEl) loadingTextEl.textContent = '얼굴을 분석하고 있습니다. 최대 60초 정도 소요될 수 있어요...';
    const progressTimer = setInterval(() => {
      elapsedMs += tickMs;
      const pct = Math.min(100, Math.floor((elapsedMs / totalMs) * 100));
      if (progressEl) progressEl.style.width = pct + '%';
      if (elapsedMs >= totalMs) clearInterval(progressTimer);
    }, tickMs);
    errorMessage.textContent = '';

    try {
      // Enforce 60s client-side timeout and allow abort
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort('timeout'), 60000);
      const res = await fetch('/api/face-analysis', { method: 'POST', body: formData, signal: controller.signal });
      clearTimeout(timeoutId);
      clearInterval(progressTimer);
      const result = await res.json();
      loadingCard.classList.add('hidden');
      if (result.status === 'success') {
        displayResults(result);
        resultCard.classList.remove('hidden');
      } else {
        errorMessage.textContent = result.message || '알 수 없는 오류가 발생했습니다.';
        uploadCard.classList.remove('hidden');
      }
    } catch (err) {
      clearInterval(progressTimer);
      loadingCard.classList.add('hidden');
      uploadCard.classList.remove('hidden');
      if (err && (err.name === 'AbortError' || err === 'timeout')) {
        errorMessage.textContent = '요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.';
      } else {
        errorMessage.textContent = '서버와 통신 중 오류가 발생했습니다.';
      }
      console.error('Analysis Error:', err);
    }
  });

  resetButton.addEventListener('click', () => {
    resultCard.classList.add('hidden');
    uploadCard.classList.remove('hidden');
    uploadForm.reset();
    fileNameSpan.textContent = '선택된 파일 없음';
    errorMessage.textContent = '';
  });
}

function displayResults(data) {
  document.getElementById('analyzed-image').src = data.vis_image;
  const reading = data.face_reading || {};
  document.getElementById('overall-reading').textContent = reading.overall_reading || '정보 없음';
  document.getElementById('personality').textContent = (reading.personality && reading.personality.characteristics) || '정보 없음';

  // Strengths list
  const strengthsList = document.getElementById('strengths-list');
  if (strengthsList) {
    strengthsList.innerHTML = '';
    const strengths = (reading.personality && reading.personality.strengths) || [];
    if (Array.isArray(strengths) && strengths.length) {
      strengths.forEach((s) => {
        const li = document.createElement('li');
        li.textContent = s;
        strengthsList.appendChild(li);
      });
      document.getElementById('strengths-section').classList.remove('hidden');
    } else {
      document.getElementById('strengths-section').classList.add('hidden');
    }
  }
  document.getElementById('advice').textContent = (reading.advice && reading.advice.general) || '정보 없음';

  const fortuneContainer = document.getElementById('fortune-aspects');
  fortuneContainer.innerHTML = '';
  if (reading.fortune_aspects) {
    for (const [key, value] of Object.entries(reading.fortune_aspects)) {
      const p = document.createElement('p');
      let keyName = key;
      if (key === 'career') keyName = '직업운';
      if (key === 'relationships') keyName = '인간관계';
      if (key === 'wealth') keyName = '재물운';
      if (key === 'health') keyName = '건강운';
      p.innerHTML = `<strong>${keyName}:</strong> ${value}`;
      fortuneContainer.appendChild(p);
    }
  }

  const luckyContainer = document.getElementById('lucky-elements');
  luckyContainer.innerHTML = '';
  if (reading.lucky) {
    if (reading.lucky.colors) {
      const p = document.createElement('p');
      p.innerHTML = `<strong>행운의 색상:</strong> ${reading.lucky.colors.join(', ')}`;
      luckyContainer.appendChild(p);
    }
    if (reading.lucky.directions) {
      const dirEl = document.getElementById('lucky-directions');
      if (dirEl) dirEl.textContent = `행운의 방향: ${reading.lucky.directions}`;
    }
    if (reading.lucky.numbers) {
      const p = document.createElement('p');
      p.innerHTML = `<strong>행운의 숫자:</strong> ${reading.lucky.numbers.join(', ')}`;
      luckyContainer.appendChild(p);
    }
  }

  // Advice list
  const adviceList = document.getElementById('advice-list');
  const adviceSection = document.getElementById('advice-section');
  if (adviceList && adviceSection) {
    adviceList.innerHTML = '';
    const specific = (reading.advice && reading.advice.specific) || [];
    if (Array.isArray(specific) && specific.length) {
      specific.forEach((tip) => {
        const li = document.createElement('li');
        li.textContent = tip;
        adviceList.appendChild(li);
      });
      adviceSection.classList.remove('hidden');
    } else {
      adviceSection.classList.add('hidden');
    }
  }

  // Landmarks coordinates
  const landmarksData = document.getElementById('landmarks-data');
  const landmarksSection = document.getElementById('landmarks-section');
  if (landmarksData && landmarksSection) {
    landmarksData.innerHTML = '';
    const landmarks = data.landmarks || [];
    if (Array.isArray(landmarks) && landmarks.length > 0) {
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(landmarks, null, 2);
      landmarksData.appendChild(pre);
      landmarksSection.classList.remove('hidden');
    } else {
      landmarksSection.classList.add('hidden');
    }
  }
}


