// Original to blis-sglang.

package sched

import "testing"

// With scheduler_recv_interval above one the loop reads its request socket only when the weights of the
// last batches reach the interval (SchedulerRecvSkipper): an idle iteration adds 1, a decode batch 1, an
// extend batch 1000. A request that arrives waits in the socket until a read; the counter starts again
// at every read.
func TestTheLoopReadsItsSocketWhenTheSkipperCountReachesTheInterval(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	cfg.SchedulerRecvInterval = 4
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	first := map[string]int{}
	for step := 0; step < 14; step++ {
		var arr []Arrival
		switch step {
		case 0:
			arr = []Arrival{{RID: "a", Prompt: []int64{1, 2, 3, 4, 5, 6}, MaxNewTokens: 20, IgnoreEOS: true}}
		case 5:
			arr = []Arrival{{RID: "b", Prompt: []int64{7, 8, 9, 10, 11, 12}, MaxNewTokens: 20, IgnoreEOS: true}}
		}
		rec, err := s.Step(arr)
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range rec.Forwards {
			if f.Mode == Extend || f.Mode == Mixed {
				for _, r := range f.RIDs {
					if _, ok := first[r]; !ok {
						first[r] = step
					}
				}
			}
		}
	}
	// a arrives at step 0 into an idle loop: three idle iterations count 1 each, the fourth reaches 4 and
	// reads. b arrives at step 5 while a decodes: the decode batches of steps 4 to 7 count 1 each (the
	// step after the extend batch reads, since 1000 reaches 4, and restarts the count), so step 8 reads.
	if first["a"] != 3 {
		t.Errorf("a's prefill ran at step %d, want 3", first["a"])
	}
	if first["b"] != 8 {
		t.Errorf("b's prefill ran at step %d, want 8", first["b"])
	}
}

// Without a skipper (an interval of one) the request is read in the iteration it arrives.
func TestAnIntervalOfOneReadsEveryIteration(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	cfg.SchedulerRecvInterval = 1
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	rec := mustStep(t, s, Arrival{RID: "a", Prompt: []int64{1, 2, 3, 4, 5, 6}, MaxNewTokens: 4, IgnoreEOS: true})
	if len(rec.Forwards) != 1 || rec.Forwards[0].RIDs[0] != "a" {
		t.Errorf("forwards %+v, want a's prefill in the arrival step", rec.Forwards)
	}
}

// Through the engine API (Enqueue then Launch) the loop is not spun while nothing runs: a request held
// with no batch to form is read at once, as SGLang's idle iterations would reach the interval within
// microseconds; a request that arrives while a batch is decoding waits for the count.
func TestTheEngineAPIReadsHeldRequestsWhenNothingRuns(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	cfg.SchedulerRecvInterval = 4
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Enqueue(Arrival{RID: "a", Prompt: []int64{1, 2, 3, 4, 5, 6}, MaxNewTokens: 3, IgnoreEOS: true}); err != nil {
		t.Fatal(err)
	}
	l, err := s.Launch()
	if err != nil || l == nil || l.Forward.RIDs[0] != "a" {
		t.Fatalf("launch %+v %v, want a's prefill at once", l, err)
	}
}
