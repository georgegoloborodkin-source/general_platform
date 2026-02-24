# Google Drive & Picker setup (fix “could not access Google”)

Use this checklist so **Sign in with Google**, **Add Google Drive folder**, and Drive sync all work.

---

## 1. Google Cloud Console — OAuth consent screen

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services** → **OAuth consent screen**.
2. Under **Scopes**, click **Add or remove scopes**.
3. Add these scopes (filter or search by name):
   - **Google Drive** → `.../auth/drive.readonly` (See and download all your Google Drive files)
   - **Gmail** → `.../auth/gmail.readonly` (Read your email messages and settings)
4. Save. If your app is in **Testing**, add your (and test users’) email under **Test users** so they can sign in.

Without these scopes, Google will not grant Drive/Gmail access and the app will not get a usable token.

---

## 2. Google Cloud Console — APIs enabled

1. Go to **APIs & Services** → **Enabled APIs & services**.
2. Enable:
   - **Google Picker API**
   - **Google Drive API**
   - **Gmail API** (if you use Gmail sync)

---

## 3. Google Cloud Console — Credentials

### OAuth 2.0 Client (you already have “Web client 1”)

- **Authorized JavaScript origins:**  
  `https://general-platform.vercel.app`  
  (no trailing slash)
- **Authorized redirect URIs:**  
  `https://slhovwkrqkmlnqcjtdfi.supabase.co/auth/v1/callback`  
  (Supabase callback — keep this)

### API key (for Picker)

1. **APIs & Services** → **Credentials** → **Create credentials** → **API key**.
2. (Optional) Restrict the key to **Google Picker API** and **Google Drive API**.
3. Copy the key — you will use it as **VITE_GOOGLE_API_KEY** on Vercel.

### New client secret (for backend token refresh)

Google no longer lets you view an existing client secret. To get a secret for your backend:

1. Open your **OAuth 2.0 Client ID** (“Web client 1”).
2. Under **Client secrets**, click **Add secret** (or **Create new secret**).
3. Copy the new secret **once** and store it securely.  
   Use it as **GOOGLE_CLIENT_SECRET** on Render (see below).

---

## 4. Supabase Dashboard

1. **Authentication** → **Providers** → **Google**:
   - **Client ID:** your OAuth client ID (e.g. `1062701746603-...apps.googleusercontent.com`).
   - **Client secret:** the same client’s secret (the original or the new one you created; Supabase only needs one that works with this client).
2. **Authentication** → **URL Configuration**:
   - **Site URL:** `https://general-platform.vercel.app`
   - **Redirect URLs:** add `https://general-platform.vercel.app/**`

Save.

---

## 5. Vercel (frontend)

In the project → **Settings** → **Environment variables**, set:

| Name                     | Value                                      | Environment  |
|--------------------------|--------------------------------------------|--------------|
| `VITE_GOOGLE_CLIENT_ID`  | Your OAuth client ID (Web client 1)       | Production   |
| `VITE_GOOGLE_API_KEY`    | The API key you created in step 3          | Production   |

Redeploy after changing env vars.

---

## 6. Render (backend)

In the backend service → **Environment** → **Environment Variables**, set:

| Name                   | Value                                |
|------------------------|--------------------------------------|
| `GOOGLE_CLIENT_ID`     | Same OAuth client ID as above       |
| `GOOGLE_CLIENT_SECRET`| The **new** client secret you created |

Save and redeploy the backend.

---

## Summary

| Issue | Fix |
|-------|-----|
| “Could not access Google” / no Drive | Add **Drive** and **Gmail** scopes on OAuth consent screen (step 1). |
| Picker never opens | Enable **Google Picker API** and set **VITE_GOOGLE_API_KEY** on Vercel (steps 2, 3, 5). |
| “Sign in again” for Drive after a while | Set **GOOGLE_CLIENT_ID** and **GOOGLE_CLIENT_SECRET** on Render (step 6) so the backend can refresh the token. |
| Redirect / wrong domain after login | Site URL and Redirect URLs in Supabase (step 4) and only use `https://general-platform.vercel.app`. |

After changing consent screen or redirect URIs, wait a few minutes and try again in an incognito window.

---

## 7. After deploying the app (one-time)

**Sign out and sign in again once** so the app can save your Google tokens for Drive/Picker. Until then, the session may not have provider tokens available for "Add Google Drive folder".
