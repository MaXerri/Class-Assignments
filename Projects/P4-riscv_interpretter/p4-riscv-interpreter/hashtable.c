#include <stdio.h>
#include <stdlib.h>
#include "linkedlist.h"
#include "hashtable.h"

struct hashtable
{
    // TODO: define hashtable struct to use linkedlist as buckets
    linkedlist_t **buckets;
    int size;
    int num_buckets;
};
/**
 * Hash function to hash a key into the range [0, max_range)
 */
static int hash(int key, int max_range)
{
    // TODO: feel free to write your own hash function (NOT REQUIRED)
    key = (key > 0) ? key : -key;
    return key % max_range;
}

hashtable_t *ht_init(int num_buckets)
{
    // TODO: create a new hashtable
    hashtable_t *table = malloc(sizeof(hashtable_t));

    table->buckets = malloc(sizeof(linkedlist_t *) * num_buckets);

    for (int i = 0; i < num_buckets; i++)
    {
        table->buckets[i] = ll_init();
    }
    table->size = 0;
    table->num_buckets = num_buckets;
    return table;
}

void ht_free(hashtable_t *table)
{
    // TODO: free a hashtable from memory

    for (int n = 0; n < table->num_buckets; n++)
    {
        ll_free(table->buckets[n]);
    }
    free(table->buckets);
    free(table);
}

void ht_add(hashtable_t *table, int key, int value)
{
    // TODO: create a new mapping from key -> value.
    // If the key already exists, replace the value.

    int hashcode = hash(key, table->num_buckets);
    int ll_size_before = ll_size(table->buckets[hashcode]);
    ll_add((table->buckets[hashcode]), key, value);
    int ll_size_after = ll_size(table->buckets[hashcode]);
    table->size = (table->size) + (ll_size_after - ll_size_before);
}

int ht_get(hashtable_t *table, int key)
{
    // TODO: retrieve the value mapped to the given key.
    // If it does not exist, return 0
    int hashcode = hash(key, table->num_buckets);
    return ll_get((table->buckets[hashcode]), key);
}

int ht_size(hashtable_t *table)
{
    // TODO: return the number of mappings in this hashtable
    return table->size;
}
