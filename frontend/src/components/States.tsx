import { useI18n } from "@/i18n";
import styles from "./States.module.css";

/** Placeholder shaped like the fixture field: ten empty columns with a scan line. */
export function FieldSkeleton() {
  const { t } = useI18n();
  return (
    <div className={styles.skeleton} role="status" aria-live="polite">
      <span className="visually-hidden">{t.state.loading}</span>
      {Array.from({ length: 10 }, (_, i) => (
        <span key={i} className={styles.column} />
      ))}
    </div>
  );
}

export function ErrorState({ onRetry }: { onRetry?: () => void }) {
  const { t } = useI18n();
  return (
    <div className={styles.message} role="alert">
      <p>{t.state.error}</p>
      {onRetry && (
        <button type="button" className={styles.button} onClick={onRetry}>
          {t.state.retry}
        </button>
      )}
    </div>
  );
}

export function EmptyState({ children }: { children: string }) {
  return (
    <div className={styles.message}>
      <p>{children}</p>
    </div>
  );
}
