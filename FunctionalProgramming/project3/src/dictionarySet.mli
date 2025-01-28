(** Sets implemented as dictionaries. *)

open Dictionary

(** The type of elements. *)
module type ElementSig = sig
  type t

  include Dictionary.KeySig with type t := t
end

(** A [Set] contains elements, which must be comparable. *)
module type Set = sig
  module Elt : ElementSig
  (** [Elt] is a module representing the type of elements in the set and
      functions on them. *)

  type elt = Elt.t
  (** [elt] is the type of elements in the set. *)

  type t
  (** [t] is the type of sets. *)

  val empty : t
  (** [empty] is the empty set. *)

  val is_empty : t -> bool
  (** [is_empty s] is [true] iff [s] is empty. *)

  val size : t -> int
  (** [size s] is the number of elements in [s]. [size empty] is [0]. *)

  val insert : elt -> t -> t
  (** [insert x s] is a set containing all the elements of [s] as well as
      element [x]. *)

  val member : elt -> t -> bool
  (** [member x s] is [true] iff [x] is an element of [s]. *)

  val remove : elt -> t -> t
  (** [remove x s] contains all the elements of [s] except [x]. If [x] is not an
      element of [s], then [remove] returns a set with the same elements as [s]. *)

  val union : t -> t -> t
  (** [union] is set union, that is, [union s1 s2] contains exactly those
      elements that are elements of [s1] or elements of [s2]. *)

  val intersect : t -> t -> t
  (** [intersect] is set intersection, that is, [intersect s1 s2] contains
      exactly those elements that are elements of [s1] and elements of [s2]. *)

  val difference : t -> t -> t
  (** [difference] is set difference, that is, [difference s1 s2] contains
      exactly those elements that are elements of [s1] but not elements of [s2]. *)

  val fold : (elt -> 'acc -> 'acc) -> 'acc -> t -> 'acc
  (** [fold f init s] is [f xn (f ... (f x1 init) ...)], if [s] contains
      [x1]..[xn]. Elements are processed in order from least to greatest, where
      [x1] is the least element and [xn] is the greatest. *)

  val to_list : t -> elt list
  (** [to_list s] is a list containing the same elements as [s]. The order of
      elements in the list is in order from the least set element to the
      greatest. *)

  include Dictionary.Stringable with type t := t
end

(** [Make] implements a set as a dictionary. The keys of the dictionary
    represent the elements of the set. The values of the dictionary are
    irrelevant. *)
module Make : functor (E : ElementSig) (DM : DictionaryMaker) ->
  Set with module Elt = E
