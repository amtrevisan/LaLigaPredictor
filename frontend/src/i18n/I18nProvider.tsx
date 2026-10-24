import { useEffect, useMemo, useState, type ReactNode } from "react";
import { I18nContext, type I18n } from "./context";
import { LOCALES, detectLocale, isLocale, type Locale } from "./locales";

const STORAGE_KEY = "matchfield.locale";

function initialLocale(): Locale {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (isLocale(stored)) return stored;
  } catch {
    // Storage can be unavailable (private mode, blocked cookies); fall back to the browser.
  }
  return detectLocale(navigator.languages ?? [navigator.language]);
}

export function I18nProvider({ children, locale: fixed }: { children: ReactNode; locale?: Locale }) {
  const [locale, setLocaleState] = useState<Locale>(() => fixed ?? initialLocale());

  useEffect(() => {
    document.documentElement.lang = LOCALES[locale].intl;
    document.title = LOCALES[locale].messages.meta.title;
    document
      .querySelector('meta[name="description"]')
      ?.setAttribute("content", LOCALES[locale].messages.meta.description);
  }, [locale]);

  const value = useMemo<I18n>(
    () => ({
      locale,
      intl: LOCALES[locale].intl,
      t: LOCALES[locale].messages,
      setLocale: (next) => {
        setLocaleState(next);
        try {
          window.localStorage.setItem(STORAGE_KEY, next);
        } catch {
          // Preference simply won't persist.
        }
      },
    }),
    [locale],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
