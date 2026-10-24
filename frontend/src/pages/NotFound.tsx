import { Link } from "react-router";
import { useI18n } from "@/i18n";
import { homeLeague } from "@/lib/leagues";
import styles from "@/views/views.module.css";

export function NotFound() {
  const { t, locale } = useI18n();
  return (
    <div className={styles.stack}>
      <h1 className={styles.viewTitle}>404</h1>
      <p>{t.state.notFound}</p>
      <p>
        <Link to={`/${homeLeague(locale)}`}>{t.state.home}</Link>
      </p>
    </div>
  );
}
