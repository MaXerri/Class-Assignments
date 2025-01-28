open Dictionary

module type ElementSig = sig
  type t

  include Dictionary.KeySig with type t := t
end

module type Set = sig
  module Elt : ElementSig

  type elt = Elt.t
  type t

  val empty : t
  val is_empty : t -> bool
  val size : t -> int
  val insert : elt -> t -> t
  val member : elt -> t -> bool
  val remove : elt -> t -> t
  val union : t -> t -> t
  val intersect : t -> t -> t
  val difference : t -> t -> t
  val fold : (elt -> 'acc -> 'acc) -> 'acc -> t -> 'acc
  val to_list : t -> elt list

  include Stringable with type t := t
end

module Unit = struct
  type t = unit

  let compare x y = EQ
  let to_string (d : t) = ""
end

module Make =
functor
  (E : ElementSig)
  (DM : DictionaryMaker)
  ->
  struct
    module Elt = E

    type elt = Elt.t

    module D = DM (Elt) (Unit)

    type t = D.t

    (*AF: The dictionary {k1: v1, ... , kn: vn} represents the set [k1; ... ;
      kn] the empty dictionary {} represents the empty set with no elements*)

    (*RI: The set does not contain duplicates and is sorted in ascending order
      with reprect to a comparison function*)

    let empty = D.empty
    let is_empty s = D.is_empty s
    let size s = D.size s
    let insert x s = D.insert x () s
    let member x s = D.member x s
    let remove x s = D.remove x s

    let to_list s =
      List.map
        (fun x ->
          match x with
          | ke, valu -> ke)
        (D.to_list s)

    let rep_ok s =
      (*Nested Helper for removing duplicates for list length comparison*)
      let remove_dup ass_lst =
        List.sort_uniq
          (fun x y ->
            match Elt.compare x y with
            | GT -> 1
            | LT -> -1
            | EQ -> 0)
          ass_lst
      in
      (*Nestde Helper to detect if the list is sorted by whatever matric
        chosen*)
      let rec is_sorted ass_lst =
        match ass_lst with
        | [] -> true
        | [ h ] -> true
        | h :: h2 :: t -> (
            match Elt.compare h h2 with
            | LT -> is_sorted (h2 :: t)
            | GT -> false
            | EQ -> false)
      in

      let u = to_list s in
      if
        is_sorted u = true (*checks if list is sorted*)
        && List.length u = List.length (remove_dup u)
        (*checks if list has duplicates*)
      then s
      else failwith "RI"
      [@@coverage off]

    let fold f init s =
      let x key _ acc = f key acc in
      D.fold x init s

    let union s1 s2 =
      fold
        (fun k init -> if member k init = true then init else insert k init)
        s1 s2

    let intersect s1 s2 =
      fold
        (fun k init -> if member k s2 = true then insert k init else init)
        empty s1

    let difference s1 s2 =
      fold
        (fun k init -> if member k s1 = true then init else insert k init)
        (fold
           (fun k init -> if member k s2 = true then init else insert k init)
           empty s1)
        s2

    let to_string s = Util.string_of_list Elt.to_string (to_list s)
  end
