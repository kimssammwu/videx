/* Host tests use the same classifier header as the S3 firmware. */
#include <assert.h>
#include <stdio.h>
#include "main/button.h"

static unsigned sample(button_state_t *state, bool raw, unsigned count, uint8_t expected)
{
    unsigned events = 0;
    for (unsigned i = 0; i < count; ++i) {
        uint8_t result = button_update(state, true, raw);
        if (result != BUTTON_RESULT_NONE) {
            assert(result == expected);
            ++events;
        }
    }
    return events;
}

int main(void)
{
    button_state_t state = {0};
    assert(button_output_flags(&state) == FRAME_FLAG_BUTTON_CLASSIFIED);
#if BUTTON_LONG_TICKS == 1 && BUTTON_DEBOUNCE_TICKS == 1
    assert(sample(&state, true, 1, BUTTON_RESULT_LONG) == 1);
    assert(sample(&state, true, 100, BUTTON_RESULT_NONE) == 0);
    assert(sample(&state, false, 1, BUTTON_RESULT_NONE) == 0);
#else
    assert(BUTTON_LONG_TICKS > BUTTON_DEBOUNCE_TICKS);
    /* Exactly one tick below LONG stays SHORT through release debounce. */
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE) == 0);
    assert(state.known && !state.pressed);
    assert(sample(&state, true, BUTTON_LONG_TICKS - 1, BUTTON_RESULT_NONE) == 0);
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_SHORT) == 1);
    assert(state.last_result == BUTTON_RESULT_SHORT && !state.pressed);
    assert(button_output_flags(&state) == 6);
    /* LONG emitted once; last result survives subsequent frames and release. */
    assert(sample(&state, true, BUTTON_LONG_TICKS, BUTTON_RESULT_LONG) == 1);
    assert(button_output_flags(&state) == 7);
    assert(sample(&state, true, 1000, BUTTON_RESULT_NONE) == 0);
    assert(state.ticks == BUTTON_LONG_TICKS);
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE) == 0);
    assert(state.last_result == BUTTON_RESULT_LONG);
    assert(sample(&state, false, 100, BUTTON_RESULT_NONE) == 0);
    assert(state.last_result == BUTTON_RESULT_LONG);

    button_reset(&state);
    /* Alternating contact chatter never creates a press. */
    for (unsigned i = 0; i < 100; ++i) {
        assert(button_update(&state, true, i % 2 != 0) == BUTTON_RESULT_NONE);
    }
    assert(!state.pressed && state.last_result == BUTTON_RESULT_NONE);
    button_reset(&state);
    assert(sample(&state, true, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE) == 0);
    /* Brief release chatter cannot split a press. */
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS - 1, BUTTON_RESULT_NONE) == 0);
    assert(state.pressed);
    assert(button_update(&state, true, true) == BUTTON_RESULT_NONE);
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_SHORT) == 1);

    button_reset(&state);
    sample(&state, true, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE);
    assert(button_update(&state, false, false) == BUTTON_RESULT_NONE);
    assert(!state.known && state.ticks == 0 && state.last_result == BUTTON_RESULT_NONE);
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE) == 0);
    sample(&state, true, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE);
    button_reset(&state); /* TCP reconnect / stream outage */
    assert(sample(&state, false, BUTTON_DEBOUNCE_TICKS, BUTTON_RESULT_NONE) == 0);
    assert(state.last_result == BUTTON_RESULT_NONE);
#endif
    puts("PASS: S3 button classifier");
    return 0;
}
