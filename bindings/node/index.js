'use strict'

// The package's entry point: the native module (loaded by the loader napi-rs generates,
// native.js), with what is simpler in JavaScript added here.

const native = require('./native.js')

const { Database, Namespace, Snapshot, Filter, Attr, attr, key } = native

/** [key, version, key, version, …] as a Map of key → version. */
function versionMap(pairs) {
  const m = new Map()
  for (let i = 0; i < pairs.length; i += 2) m.set(pairs[i], pairs[i + 1])
  return m
}

for (const C of [Namespace, Snapshot]) {
  Object.defineProperty(C.prototype, 'versions', {
    value: function versions() {
      return versionMap(this._versions())
    },
    writable: true,
    configurable: true,
  })
}

/** Any iterable (a generator, a Set) as an array; anything else as it is, for the native check. */
function list(items) {
  if (Array.isArray(items) || items == null || typeof items === 'string' || typeof items[Symbol.iterator] !== 'function') return items
  return Array.from(items)
}

const { upsertMany, deleteMany, apply } = Namespace.prototype
Object.defineProperties(Namespace.prototype, {
  upsertMany: { value: function (docs) { return upsertMany.call(this, list(docs)) }, writable: true, configurable: true },
  deleteMany: { value: function (keys) { return deleteMany.call(this, list(keys)) }, writable: true, configurable: true },
  apply: {
    value: function (changes) {
      if (changes !== null && typeof changes === 'object' && !Array.isArray(changes)) {
        changes = { ...changes }
        if ('upsert' in changes) changes.upsert = list(changes.upsert)
        if ('delete' in changes) changes.delete = list(changes.delete)
      }
      return apply.call(this, changes)
    },
    writable: true,
    configurable: true,
  },
})

// `using ns = db.namespace(...)` closes it at the end of the block.
if (typeof Symbol.dispose === 'symbol') {
  for (const C of [Database, Namespace, Snapshot]) {
    Object.defineProperty(C.prototype, Symbol.dispose, { value: function () { this.close() }, configurable: true })
  }
}

const inspect = Symbol.for('nodejs.util.inspect.custom')
for (const C of [Filter, Attr]) {
  Object.defineProperty(C.prototype, inspect, { value: function () { return this.toString() }, configurable: true })
}
Object.defineProperty(Namespace.prototype, inspect, {
  value: function () { return this.closed ? `Namespace(${JSON.stringify(this.name)}, closed)` : `Namespace(${JSON.stringify(this.name)})` },
  configurable: true,
})

module.exports = { Database, Namespace, Snapshot, Filter, Attr, attr, key }
