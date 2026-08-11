import { useCallback, useRef, useState } from 'react';
import Header from './components/Header/Header';
import ResizeHandle from './components/common/ResizeHandle';
import useResizablePanel from './hooks/useResizablePanel';
import Footer from './components/Footer/Footer';
import ChatPane from './components/Chat/ChatPane';
import MicButton from './components/Chat/MicButton';
import AssistantIndicator from './components/Chat/AssistantIndicator';
import { LiveQueueFeed } from './components/Chat/LiveQueueFeed';
import { PerformanceDashboard } from './components/Dashboard/PerformanceDashboard';
import useMicDevices from './hooks/useMicDevices';
import { useKpis } from './hooks/useKpis';
import { useVoiceSession } from './hooks/useVoiceSession';
import { useMetrics } from './hooks/useMetrics';

export default function App() {
  const { devices, selectedId, setSelectedId, error: micError } = useMicDevices();
  const { kpis, refresh: refreshKpis } = useKpis();
  const { metrics } = useMetrics();
  const [ingestBusy, setIngestBusy] = useState(false);

  const onTurnComplete = useCallback(() => {
    refreshKpis();
  }, [refreshKpis]);

  const {
    phase,
    messages,
    partialUser,
    partialAssistant,
    statusText,
    playbackState,
    conversationMode,
    startConversation,
    endConversation,
    interruptSpeaking,
  } = useVoiceSession({ deviceId: selectedId, enabled: !ingestBusy, onTurnComplete });

  const orderActive = phase === 'listening' || phase === 'processing' || playbackState !== 'idle';

  // ── Resizable chat / dashboard split ──────────────────────────────────────
  const { percent, setPercent, reset } = useResizablePanel();
  const splitRef = useRef<HTMLDivElement>(null);

  // During a drag the grid template is written straight to the DOM node. Going
  // through React state here would re-render the recharts hardware graphs on
  // every pointermove and visibly stutter; the value is committed once on
  // pointerup (see ResizeHandle).
  const previewSplit = useCallback((next: number) => {
    const el = splitRef.current;
    if (!el) return;
    el.style.setProperty('--split-chat', `${next}%`);
  }, []);

  return (
    <div className="flex flex-col h-full bg-gray-100 font-text">
      <Header />

      {/*
        Responsive layout:
          • Mobile/tablet (<lg): single column — chat on top, dashboard below.
            The ResizeHandle is hidden here; a horizontal split is meaningless
            when the panes are stacked.
          • Desktop (lg+):       two columns separated by a draggable divider.
            The chat column width comes from the `--split-chat` custom property
            so a drag can update it without a React re-render; `percent` keeps
            it correct across renders and reloads.
        Padding is kept tight (p-3) so panels reach near screen edges.
        No maxWidth / mx-auto so the layout fills the full viewport width.
      */}
      <main className="flex-1 min-h-0">
        <div
          ref={splitRef}
          className="h-full p-3 flex flex-col gap-3 lg:grid lg:gap-0"
          style={
            {
              '--split-chat': `${percent}%`,
              gridTemplateColumns: 'var(--split-chat) 14px minmax(0, 1fr)',
              gridTemplateRows: '1fr',
            } as React.CSSProperties
          }
        >
          {/* ── Chat pane ─────────────────────────────────────────────────── */}
          <section
            className="flex flex-col bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm
                       min-h-[420px] lg:min-h-0 min-w-0"
          >
            {/* Container 1 — fixed-height live queue video. Demo screen
                shows the QSR people-queue feed above the chat at all times.
                The QSR tab on the right (QsrPanel) keeps its own copy. */}
            <div className="shrink-0 p-2">
              <div className="overflow-hidden rounded-lg border border-gray-200 shadow-sm">
                <LiveQueueFeed height="220px" />
              </div>
            </div>

            {/* Divider between the two containers */}
            <div className="mx-3 h-px shrink-0 bg-gray-300" />

            {/* Container 2 — chat + mic controls */}
            <div className="flex flex-1 min-h-0 flex-col p-2">
              <div className="flex flex-1 min-h-0 flex-col overflow-hidden rounded-lg border border-gray-200 shadow-sm">
                <ChatPane
                  messages={messages}
                  partialUser={partialUser}
                  partialAssistant={partialAssistant}
                  phase={phase}
                />
                <div className="shrink-0 border-t border-gray-200 bg-gray-50/80 px-4 sm:px-6 py-3 sm:py-4">
                  <div className="flex flex-col items-center gap-2 sm:gap-3">
                    <AssistantIndicator phase={phase} playbackState={playbackState} />
                    <MicButton
                      phase={phase}
                      playbackState={playbackState}
                      locked={ingestBusy}
                      conversationMode={conversationMode}
                      onStart={startConversation}
                      onStop={endConversation}
                      onInterrupt={interruptSpeaking}
                    />
                    <p className="text-xs text-kiosk-textlo text-center min-h-[1rem] max-w-sm">
                      {statusText}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ── Draggable divider (desktop only) ──────────────────────────── */}
          <ResizeHandle
            percent={percent}
            onCommit={setPercent}
            onReset={reset}
            onPreview={previewSplit}
            containerRef={splitRef}
          />

          {/* ── Performance & Settings — direct grid item, fills its column ── */}
          <PerformanceDashboard
            kpis={kpis}
            metrics={metrics}
            phase={phase}
            orderActive={orderActive}
            devices={devices}
            selectedDeviceId={selectedId}
            onSelectDevice={setSelectedId}
            micError={micError}
            onIngestStateChange={setIngestBusy}
            onRefreshKpis={refreshKpis}
          />
        </div>
      </main>

      <Footer />
    </div>
  );
}
