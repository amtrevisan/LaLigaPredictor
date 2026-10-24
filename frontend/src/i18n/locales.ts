import de from "./locales/de";
import en from "./locales/en";
import es from "./locales/es";
import fr from "./locales/fr";
import it from "./locales/it";
import pt from "./locales/pt";

export const LOCALES = {
  en: { label: "English", messages: en, intl: "en-GB" },
  es: { label: "Español", messages: es, intl: "es-ES" },
  it: { label: "Italiano", messages: it, intl: "it-IT" },
  de: { label: "Deutsch", messages: de, intl: "de-DE" },
  fr: { label: "Français", messages: fr, intl: "fr-FR" },
  pt: { label: "Português", messages: pt, intl: "pt-BR" },
} as const;

export type Locale = keyof typeof LOCALES;

export function isLocale(value: string | null | undefined): value is Locale {
  return value != null && value in LOCALES;
}

export function detectLocale(languages: readonly string[]): Locale {
  for (const tag of languages) {
    const base = tag.toLowerCase().split("-")[0];
    if (isLocale(base)) return base;
  }
  return "en";
}
