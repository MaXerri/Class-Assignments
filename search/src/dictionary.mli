(** Maps from keys to values *)

(** [order] represents the concepts of strictly-less-than, equal-to, and
    strictly greater-than. *)
type order =
  | LT
  | EQ
  | GT

(** A [Comparable] is a value that can be compared. The comparison is a total
    order on the values. *)
module type Comparable = sig
  type t
  (** The type of comparable values. *)

  val compare : t -> t -> order
  (** [compare t1 t2] is [LT] if [t1] is less than [t2], [EQ] if [t1] is equal
      to [t2], or [GT] if [t1] is greater than [t2]. *)
end

(** A [Stringable] is a value that can be converted to a string. *)
module type Stringable = sig
  type t
  (** The type of stringable values. *)

  val to_string : t -> string
  (** [to_string v] is a textual representation of [v]. *)
end

(** A module that matches [KeySig] is suitable for use as the type of keys in a
    [Dictionary]. Keys must be comparable and stringable. *)
module type KeySig = sig
  type t

  include Comparable with type t := t
  include Stringable with type t := t
end

(** A module that matches [ValueSig] is suitable for use as the type of values
    in a [Dictionary]. Values must be stringable. *)
module type ValueSig = sig
  type t

  include Stringable with type t := t
end

(** A [Dictionary] maps keys to values. *)
module type Dictionary = sig
  module Key : KeySig
  (** [Key] is a module representing the type of keys in the dictionary and
      functions on them. *)

  module Value : ValueSig
  (** [Value] is a module representing the type of values in the dictionary and
      functions on them. *)

  type key = Key.t
  (** [key] is the type of keys in the dictionary. *)

  type value = Value.t
  (** [value] is the type of values in the dictionary. *)

  type t
  (** [t] is the type of dictionaries. *)

  val empty : t
  (** [empty] is the empty dictionary *)

  val is_empty : t -> bool
  (** [is_empty d] is [true] iff [d] is empty. *)

  val size : t -> int
  (** [size d] is the number of bindings in [d]. [size empty] is [0]. *)

  val insert : key -> value -> t -> t
  (** [insert k v d] is [d] with [k] bound to [v]. If [k] was already bound, its
      previous value is replaced with [v]. *)

  val member : key -> t -> bool
  (** [member k d] is [true] iff [k] is bound in [d]. *)

  val find : key -> t -> value option
  (** [find k d] is [Some v] if [k] is bound to [v] in [d]; or if [k] is not
      bound, then it is [None]. *)

  val remove : key -> t -> t
  (** [remove k d] contains all the bindings of [d] except a binding for [k]. If
      [k] is not bound in [d], then [remove] returns a dictionary with the same
      bindings as [d]. *)

  val fold : (key -> value -> 'acc -> 'acc) -> 'acc -> t -> 'acc
  (** [fold f init d] is [f kn vn (f ... (f k1 v1 init) ...)], if [d] binds [ki]
      to [vi]. Bindings are processed in order from least to greatest, where
      [k1] is the least key and [kn] is the greatest. *)

  val to_list : t -> (key * value) list
  (** [to_list d] is an association list containing the same bindings as [d].
      The elements in the list are in order from the least key to the greatest.
      There are no duplicate keys in the list. *)

  include Stringable with type t := t
end

(** A [DictionaryMaker] is a functor that makes a [Dictionary] out of modules
    representing keys and values. *)
module type DictionaryMaker = functor (K : KeySig) (V : ValueSig) ->
  Dictionary with module Key = K and module Value = V
