import { createContext, useContext } from "react";
import type { Locale } from "./locales";
import type { Messages } from "./locales/en";

export interface I18n {
  locale: Locale;
  intl: string;
  t: Messages;
  setLocale: (locale: Locale) => void;
}

export const I18nContext = createContext<I18n | null>(null);

export function useI18n(): I18n {
  const context = useContext(I18nContext);
  if (!context) throw new Error("useI18n must be used inside <I18nProvider>");
  return context;
}
