#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "cache.h"
#include "print_helpers.h"

cache_t *make_cache(int capacity, int block_size, int assoc, enum protocol_t protocol, bool lru_on_invalidate_f){
  cache_t *cache = malloc(sizeof(cache_t));
  cache->stats = make_cache_stats();
  
  cache->capacity = capacity;      // in Bytes
  cache->block_size = block_size;  // in Bytes
  cache->assoc = assoc;            // 1, 2, 3... etc.

  // FIX THIS CODE!
  // first, correctly set these 5 variables. THEY ARE ALL WRONG
  // note: you may find math.h's log2 function useful
  cache->n_cache_line = capacity / block_size;
  cache->n_set = cache->n_cache_line / assoc;
  cache->n_offset_bit = log2(block_size);
  cache->n_index_bit = log2(cache->n_set);
  cache->n_tag_bit = ADDRESS_SIZE - cache->n_index_bit - cache->n_offset_bit;

  // next create the cache lines and the array of LRU bits
  // - malloc an array with n_rows
  // - for each element in the array, malloc another array with n_col
  // FIX THIS CODE!

  cache->lines = malloc(sizeof(cache_line_t*) * cache->n_set);
  for(int i = 0; i < cache->n_set; i++)
      cache->lines[i] = malloc(sizeof(cache_line_t) * assoc);
  cache->lru_way = malloc(sizeof(int) * cache->n_set);

  // initializes cache tags to 0, dirty bits to false,
  // state to INVALID, and LRU bits to 0
  // FIX THIS CODE!
  for (int i = 0; i < 1; i++) {
    cache->lru_way[i] = 0;
    for (int j = 0; j < 1; j++) {
      cache->lines[i][j].tag = 0;
      cache->lines[i][j].dirty_f = false;
      cache->lines[i][j].state = INVALID;
    }
  }

  cache->protocol = protocol;
  cache->lru_on_invalidate_f = lru_on_invalidate_f;
  
  return cache;
}

/* Given a configured cache, returns the tag portion of the given address.
 *
 * Example: a cache with 4 bits each in tag, index, offset
 * in binary -- get_cache_tag(0b111101010001) returns 0b1111
 * in decimal -- get_cache_tag(3921) returns 15 
 */
unsigned long get_cache_tag(cache_t *cache, unsigned long addr) {
  // FIX THIS CODE!
  return addr >> (cache->n_index_bit + cache->n_offset_bit);
}

/* Given a configured cache, returns the index portion of the given address.
 *
 * Example: a cache with 4 bits each in tag, index, offset
 * in binary -- get_cache_index(0b111101010001) returns 0b0101
 * in decimal -- get_cache_index(3921) returns 5
 */
unsigned long get_cache_index(cache_t *cache, unsigned long addr) {
  // FIX THIS CODE!
  return (addr >> cache->n_offset_bit) % (1 << cache->n_index_bit);
}

/* Given a configured cache, returns the given address with the offset bits zeroed out.
 *
 * Example: a cache with 4 bits each in tag, index, offset
 * in binary -- get_cache_block_addr(0b111101010001) returns 0b111101010000
 * in decimal -- get_cache_block_addr(3921) returns 3920
 */
unsigned long get_cache_block_addr(cache_t *cache, unsigned long addr) {
  // FIX THIS CODE!
  return (addr >> cache->n_offset_bit) << cache->n_offset_bit;
}


/* this method takes a cache, an address, and an action
 * it proceses the cache access. functionality in no particular order: 
 *   - look up the address in the cache, determine if hit or miss
 *   - update the LRU_way, cacheTags, state, dirty flags if necessary
 *   - update the cache statistics (call update_stats)
 * return true if there was a hit, false if there was a miss
 * Use the "get" helper functions above. They make your life easier.
 */
bool access_cache(cache_t *cache, unsigned long addr, enum action_t action) {
  // FIX THIS CODE!
  unsigned long tag = get_cache_tag(cache, addr);
  unsigned long index = get_cache_index(cache, addr);
  cache_line_t *line = NULL;

  // check for whether it is hit/miss, is upgrade miss, and is writeback
  bool hit_f = false;
  bool upgrade_miss_f = false;
  bool writeback_f = false;

  log_set(index);

  // check if hit or miss
  for (int i = 0; i < cache->assoc; i++) {
    line = &(cache->lines[index][i]);
    if (line->tag == tag && line->state != INVALID) {
      log_way(i);
      hit_f = true;
      
      // update LRU
      if (action == LOAD || action == STORE) {
        cache->lru_way[index] = (i + 1) % cache->assoc;
      }
      break;
    }
  }

  // load cache block according to LRU if is not hit
  if (!hit_f) {
    line = &(cache->lines[index][cache->lru_way[index]]);
    if (action == LOAD || action == STORE) {
      cache->lru_way[index] += 1;
      cache->lru_way[index] %= cache->assoc;
    }
  }

  // check for each protocol (NONE, VI, MSI)
  // for each protocol, check whether it is hit or miss
  // afterwards, check state and action to update
  //  current line's dirty_f, tag, state, and writeback_f, upgrade_miss_f, hit_f
  // when it's not hit, we evict back to what the previous
  if (cache->protocol == NONE) {
    if (hit_f) {
      if (action == STORE) {
        line->dirty_f = true;
      }
    } else {
      if (action == STORE) {
        line->dirty_f = true;
        line->tag = tag;
        line->state = VALID;
        writeback_f = true;
      } else if (action == LOAD) {
        line->dirty_f = false;
        line->tag = tag;
        line->state = VALID;
        writeback_f = false;
      }
    }
  } else if (cache->protocol == VI) {
    if (hit_f) {
      if (line->state == VALID) {
        line->dirty_f = true;
      } else if (action == LD_MISS || action == ST_MISS) {
        line->state = INVALID;
        writeback_f = line->dirty_f;
      } else {
        if (action == STORE) {
          line->dirty_f = true;
          line -> tag = tag;
          line -> state = VALID;
        } else if (action == LOAD) {
          line->dirty_f = false;
          line -> tag = tag;
          line -> state = VALID;
        }
      }
    } else {
      if (action == STORE) {
        line->dirty_f = true;
        line -> tag = tag;
        line -> state = VALID;
        writeback_f = line -> state == VALID;
      } else if (action == LOAD) {
        line->dirty_f = false;
        line -> tag = tag;
        line -> state = VALID;
        writeback_f = false;
      }
    }
  } else {
    if (hit_f) {
      if (line->state == INVALID) {
        if (action == STORE) {
          line->dirty_f = true;
          line->tag = tag;
          line->state = MODIFIED;
        } else if (action == LOAD) {
          line->dirty_f = false;
          line->tag = tag;
          line->state = SHARED;
        }
      } else if (line->state == MODIFIED) {
        if (action == STORE) {
          line->dirty_f = true;
        } else if (action == LD_MISS) {
          line -> dirty_f = false;
          line -> state = SHARED;
          writeback_f = true;
        } else if (action == ST_MISS) {
          line -> dirty_f = false;
          line -> state = INVALID;
          writeback_f = true;
        }
      } else {
        if (action == ST_MISS) {
          line->state = INVALID;
        } else if (action == STORE) {
          line->dirty_f = true;
          line->state = MODIFIED;
          hit_f = false;
          upgrade_miss_f = true;
        }
      }
    } else {
      if (action == STORE) {
        line -> dirty_f = true;
        line -> tag = tag;
        line -> state = MODIFIED;
        writeback_f = line -> state == VALID;
      } else if (action == LOAD) {
        line -> dirty_f = false;
        line -> tag = tag;
        line -> state = SHARED;
        writeback_f = false;
      }
    }
  }

  // update stats after each cache access
  update_stats(cache->stats, hit_f, writeback_f, upgrade_miss_f, action);
  return hit_f;
}
