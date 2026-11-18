// Original to blis-sglang.

package sglangengine

import (
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// runReport is what a scenario run makes observable: per-request completion, first-token and
// inter-token times on the virtual clock, the KV pool's size and peak use. It is the part of a simulator's Result an engine alone determines.
type runReport struct {
	Finished   map[string]engine.Finished
	FirstToken map[string]engine.Time
	Done       map[string]engine.Time
	Emits      map[string][]engine.Time // when each output token became visible
	PeakKVUsed int64
	KVTotal    int64
	Order      []string
	End        engine.Time
}

// runScenario drives an engine built from the document block through the factory the way a
// simulator does: requests are added at their arrival, a batch is scheduled whenever the engine
// is idle, priced, and completed at its end.
func runScenario(t *testing.T, block string, reqs []engine.Request) runReport {
	t.Helper()
	e, err := FactoryFor(newHost()).New(specOf(t, block))
	if err != nil {
		t.Fatal(err)
	}
	rep := runReport{Finished: map[string]engine.Finished{}, FirstToken: map[string]engine.Time{},
		Done: map[string]engine.Time{}, Emits: map[string][]engine.Time{}}
	pending := append([]engine.Request(nil), reqs...)
	sort.SliceStable(pending, func(i, j int) bool { return pending[i].ArrivalTime < pending[j].ArrivalTime })
	var now engine.Time
	for guard := 0; len(rep.Finished) < len(reqs); guard++ {
		if guard > 10000 {
			t.Fatalf("the run did not finish: %d of %d requests done at t=%d", len(rep.Finished), len(reqs), now)
		}
		for len(pending) > 0 && pending[0].ArrivalTime <= now {
			if err := e.Add(pending[0], now); err != nil {
				t.Fatal(err)
			}
			pending = pending[1:]
		}
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			if len(pending) == 0 {
				t.Fatalf("the engine is idle at t=%d with %d requests unfinished", now, len(reqs)-len(rep.Finished))
			}
			now = pending[0].ArrivalTime
			continue
		}
		end := now + e.StepTime(b)
		res, err := e.Complete(b.ID, engine.Outcome{}, end)
		if err != nil {
			t.Fatal(err)
		}
		for id, n := range res.Emitted {
			for i := 0; i < n; i++ {
				rep.Emits[id] = append(rep.Emits[id], end)
			}
		}
		for _, id := range res.FirstTokens {
			rep.FirstToken[id] = end
		}
		for _, f := range res.Finished {
			rep.Finished[f.ID] = f
			rep.Done[f.ID] = end
			rep.Order = append(rep.Order, f.ID)
		}
		st := e.(interface{ Snapshot() engine.State }).Snapshot()
		rep.KVTotal = st.KVBlocksTotal
		if st.KVBlocksUsed > rep.PeakKVUsed {
			rep.PeakKVUsed = st.KVBlocksUsed
		}
		now = end
	}
	rep.End = now
	return rep
}

func scenarioRequests() []engine.Request {
	shared := make([]int32, 192)
	for i := range shared {
		shared[i] = int32(1000 + i)
	}
	tail := func(seed int32, n int) []int32 {
		p := append([]int32(nil), shared...)
		for i := 0; i < n; i++ {
			p = append(p, seed*100+int32(i))
		}
		return p
	}
	out := func(n int) []int32 {
		o := make([]int32, n)
		for i := range o {
			o[i] = int32(7 + i)
		}
		return o
	}
	return []engine.Request{
		{ID: "a", Prompt: tail(1, 64), Output: out(6), MaxOutputTokens: 6, ArrivalTime: 0},
		{ID: "b", Prompt: tail(2, 80), Output: out(4), MaxOutputTokens: 4, ArrivalTime: 0},
		// c arrives after a finished, so its 192-token shared prefix is served from the cache.
		{ID: "c", Prompt: tail(3, 32), Output: out(5), MaxOutputTokens: 5, ArrivalTime: 200_000},
	}
}

// A scenario of the sglang kind runs to completion and every request leaves a result: it
// finishes for the reason the scenario states, its first token precedes its last, its tokens are
// spaced by the engine's passes, and the KV pool was used within its size. The same
// input run twice gives the identical report (INV-13).
func TestSGLangScenarioRunsEndToEnd(t *testing.T) {
	const block = "model_path: m\nmax_running_requests: 8\nchunked_prefill_size: 128\n"
	reqs := scenarioRequests()
	rep := runScenario(t, block, reqs)

	want := map[string]int{"a": 6, "b": 4, "c": 5}
	for id, n := range want {
		f, ok := rep.Finished[id]
		if !ok {
			t.Fatalf("request %s never finished", id)
		}
		if f.OutputTokens != n || f.Reason != engine.FinishStop {
			t.Errorf("%s finished %+v, want %d output tokens and reason stop", id, f, n)
		}
		if got := len(rep.Emits[id]); got != n {
			t.Errorf("%s emitted %d tokens, want %d", id, got, n)
		}
		ttft := rep.FirstToken[id] - reqs[idx(reqs, id)].ArrivalTime
		if ttft < 5000 {
			t.Errorf("%s TTFT %d us is under one pass (5000 us)", id, ttft)
		}
		// Every pass costs the fixed 5 ms of the fake cost model, so a decode token follows the
		// previous one by one pass, or by more when a prefill pass of another request ran between.
		for i := 1; i < len(rep.Emits[id]); i++ {
			if itl := rep.Emits[id][i] - rep.Emits[id][i-1]; itl < 5000 {
				t.Errorf("%s inter-token latency %d us is under one pass", id, itl)
			}
		}
		if rep.Done[id] != rep.Emits[id][len(rep.Emits[id])-1] {
			t.Errorf("%s finished at %d us, its last token was visible at %d us", id, rep.Done[id], rep.Emits[id][len(rep.Emits[id])-1])
		}
	}
	// c has a 192-token prefix in the cache when it arrives (a and b are done): its first pass
	// computes only the uncached part, so its TTFT is one pass, the same as a lone short prompt.
	if ttft := rep.FirstToken["c"] - 200_000; ttft != 5000 {
		t.Errorf("c TTFT %d us, want one pass (5000 us) with its shared prefix served from the cache", ttft)
	}
	if rep.KVTotal == 0 || rep.PeakKVUsed == 0 || rep.PeakKVUsed > rep.KVTotal {
		t.Errorf("KV pool figures: peak used %d of %d blocks", rep.PeakKVUsed, rep.KVTotal)
	}

	again := runScenario(t, block, scenarioRequests())
	if !reflect.DeepEqual(rep, again) {
		t.Errorf("the same scenario run twice gave different reports:\n%+v\n%+v", rep, again)
	}
}

func idx(reqs []engine.Request, id string) int {
	for i, r := range reqs {
		if r.ID == id {
			return i
		}
	}
	return -1
}
