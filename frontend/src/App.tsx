import { Component, type ErrorInfo, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router";
import { Shell } from "@/components/Shell";
import { ErrorState } from "@/components/States";
import { useI18n } from "@/i18n";
import { homeLeague } from "@/lib/leagues";
import { LeaguePage } from "@/pages/LeaguePage";
import { NotFound } from "@/pages/NotFound";
import { ModelView } from "@/views/ModelView";

class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Unhandled render error", error, info.componentStack);
  }

  render() {
    return this.state.failed ? <ErrorState onRetry={() => window.location.reload()} /> : this.props.children;
  }
}

/** Remounts the boundary per route, so navigating away from a crashed view recovers. */
function RouteBoundary({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  return <ErrorBoundary key={pathname}>{children}</ErrorBoundary>;
}

function HomeRedirect() {
  const { locale } = useI18n();
  return <Navigate to={`/${homeLeague(locale)}`} replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <RouteBoundary>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<HomeRedirect />} />
            <Route path="model" element={<ModelView />} />
            <Route path=":leagueId" element={<LeaguePage view="fixtures" />} />
            <Route path=":leagueId/results" element={<LeaguePage view="results" />} />
            <Route path=":leagueId/table" element={<LeaguePage view="table" />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </RouteBoundary>
    </BrowserRouter>
  );
}
