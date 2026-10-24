import { NavLink, Outlet, useLocation, useParams } from "react-router";
import { LOCALES, isLocale, useI18n } from "@/i18n";
import { useManifest } from "@/lib/api";
import { dateTime, fill } from "@/lib/format";
import { homeLeague, isLeagueId, type LeagueId } from "@/lib/leagues";
import { useLastLeague } from "@/lib/useLastLeague";
import { ArrowUpRight } from "./icons";
import { LogoMark } from "./Logo";
import styles from "./Shell.module.css";

export const SOURCE_URL = "https://github.com/amtrevisan/LaLigaPredictor";

const AUTHOR = "Alex Morales Trevisan";
const AUTHOR_LINKS = [
  { label: "LinkedIn", href: "https://www.linkedin.com/in/alex-morales-dev/" },
  { label: "GitHub", href: "https://github.com/AlexMoralesDev" },
  { label: "YouTube", href: "https://www.youtube.com/@alexmoralesdev" },
  { label: "Instagram", href: "https://www.instagram.com/alexmoralesdev" },
  { label: "TikTok", href: "https://www.tiktok.com/@alexmoralesdev" },
];

export function Shell() {
  const { t, intl, locale, setLocale } = useI18n();
  const params = useParams();
  const manifest = useManifest();
  const remembered = useLastLeague(isLeagueId(params.leagueId) ? params.leagueId : undefined);
  const league: LeagueId = remembered ?? homeLeague(locale);
  const inLeague = isLeagueId(params.leagueId);
  // Switching league keeps the current view (fixtures, results or table).
  const view = useLocation().pathname.split("/")[2];

  const tabs = [
    { to: `/${league}`, label: t.nav.fixtures, end: true },
    { to: `/${league}/results`, label: t.nav.results, end: false },
    { to: `/${league}/table`, label: t.nav.table, end: false },
    { to: "/model", label: t.nav.model, end: false },
  ];

  return (
    <div className={styles.page}>
      <a className={styles.skip} href="#main">
        {t.nav.skip}
      </a>
      <header className={styles.header}>
        <NavLink to={`/${league}`} className={styles.wordmark} aria-label="Oddsless">
          <LogoMark />
          <span aria-hidden="true">ODDSLESS</span>
        </NavLink>
        <nav className={styles.tabs} aria-label={t.nav.sections}>
          {tabs.map((tab) => (
            <NavLink key={tab.to} to={tab.to} end={tab.end} className={styles.tab}>
              {tab.label}
            </NavLink>
          ))}
        </nav>
        <div className={styles.meta}>
          {manifest.data && (
            <span className={styles.asOf}>{fill(t.nav.updated, { time: dateTime(manifest.data.dataAsOf, intl) })}</span>
          )}
          <label className={styles.language}>
            <span className="visually-hidden">{t.nav.language}</span>
            <select
              value={locale}
              onChange={(event) => isLocale(event.target.value) && setLocale(event.target.value)}
            >
              {Object.entries(LOCALES).map(([code, { label }]) => (
                <option key={code} value={code}>
                  {code.toUpperCase()} · {label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {inLeague && manifest.data && (
        <nav className={styles.leagues} aria-label={t.nav.leagues}>
          {manifest.data.leagues.map((item) => (
            <NavLink
              key={item.id}
              to={view ? `/${item.id}/${view}` : `/${item.id}`}
              className={styles.league}
            >
              {item.name}
            </NavLink>
          ))}
        </nav>
      )}

      <main id="main" className={styles.main}>
        <Outlet />
      </main>

      <footer className={styles.footer}>
        <div className={styles.signature}>
          <p className={styles.author}>
            <span>{t.footer.builtBy}</span>
            <strong>{AUTHOR}</strong>
          </p>
          <ul className={styles.social} aria-label={AUTHOR}>
            {AUTHOR_LINKS.map((link) => (
              <li key={link.label}>
                <a href={link.href} rel="noreferrer me" target="_blank">
                  {link.label}
                  <ArrowUpRight />
                </a>
              </li>
            ))}
          </ul>
        </div>
        <p className={styles.credits}>
          <a href="https://www.football-data.co.uk" rel="noreferrer">
            {t.footer.data}
          </a>
          <span aria-hidden="true"> · </span>
          <a href="https://www.thesportsdb.com" rel="noreferrer">
            {t.footer.images}
          </a>
          <span aria-hidden="true"> · </span>
          <a href={SOURCE_URL} rel="noreferrer">
            {t.footer.source}
          </a>
        </p>
        <p className={styles.disclaimer}>{t.footer.disclaimer}</p>
      </footer>
    </div>
  );
}
