#ifndef PROXY_BUTTON_H
#define PROXY_BUTTON_H

#include <stdbool.h>
#include <stdint.h>

/* One complete LEFT frame received by S3 is one tick (roughly 1 s at 30 FPS). */
#ifndef BUTTON_LONG_TICKS
#define BUTTON_LONG_TICKS 30U
#endif
#ifndef BUTTON_DEBOUNCE_TICKS
#define BUTTON_DEBOUNCE_TICKS 3U
#endif
_Static_assert(BUTTON_DEBOUNCE_TICKS >= 1 &&
               BUTTON_LONG_TICKS >= BUTTON_DEBOUNCE_TICKS &&
               BUTTON_LONG_TICKS <= UINT32_MAX, "Invalid button tick thresholds");

#define FRAME_FLAG_BUTTON_PRESSED (1U << 0)
#define FRAME_FLAG_BUTTON_VALID (1U << 1)
#define FRAME_FLAG_BUTTON_CLASSIFIED (1U << 2)

enum {
    BUTTON_RESULT_NONE = 0,
    BUTTON_RESULT_SHORT = 1,
    BUTTON_RESULT_LONG = 2,
};

typedef struct {
    bool known;
    bool pressed;
    bool candidate;
    bool long_reported;
    uint32_t candidate_ticks;
    uint32_t ticks;
    uint8_t last_result;
} button_state_t;

static inline void button_reset(button_state_t *state)
{
    *state = (button_state_t){0};
}

/* No clock reads, delays, or allocation. Returns an event once per press.
 * Keep last_result in every outgoing frame so USB queue drops don't erase it. */
static inline uint8_t button_update(button_state_t *state, bool valid, bool raw)
{
    if (!valid) {
        button_reset(state);
        return BUTTON_RESULT_NONE;
    }
    uint8_t event = BUTTON_RESULT_NONE;
    if (state->candidate_ticks != 0 && raw == state->candidate) {
        if (state->candidate_ticks < BUTTON_DEBOUNCE_TICKS) {
            ++state->candidate_ticks;
        }
    } else {
        state->candidate = raw;
        state->candidate_ticks = 1;
    }
    if (state->candidate_ticks >= BUTTON_DEBOUNCE_TICKS) {
        state->known = true;
        if (raw != state->pressed) {
            state->pressed = raw;
            if (raw) {
                state->ticks = BUTTON_DEBOUNCE_TICKS;
                state->long_reported = false;
            } else {
                if (!state->long_reported) {
                    event = BUTTON_RESULT_SHORT;
                    state->last_result = event;
                }
                state->ticks = 0;
            }
        } else if (state->pressed && raw && state->ticks < BUTTON_LONG_TICKS) {
            ++state->ticks;
        }
    } else if (state->pressed && raw && state->ticks < BUTTON_LONG_TICKS) {
        ++state->ticks;
    }
    /* Freeze length during pending release; release bounce cannot make LONG. */
    if (state->pressed && state->ticks >= BUTTON_LONG_TICKS && !state->long_reported) {
        state->long_reported = true;
        event = BUTTON_RESULT_LONG;
        state->last_result = event;
    }
    return event;
}

static inline uint8_t button_output_flags(const button_state_t *state)
{
    return FRAME_FLAG_BUTTON_CLASSIFIED |
           (state->known ? FRAME_FLAG_BUTTON_VALID : 0) |
           (state->pressed ? FRAME_FLAG_BUTTON_PRESSED : 0);
}

#endif
