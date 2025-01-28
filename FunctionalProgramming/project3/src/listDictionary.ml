open Dictionary

module Make : DictionaryMaker =
functor
  (K : KeySig)
  (V : ValueSig)
  ->
  struct
    module Key = K
    module Value = V

    type key = K.t
    type value = V.t
    type t = (key * value) list

    (*AF: The association list [(k1,v1);...;(kn,vn)] represents the dictionary
      which maps keys k1,...,kn to their correpsonding values v1,...,vn which is
      {k1: v1, ... ,kn: vn}. The empty association list [] represents the empty
      dictionary with no keys *)

    (*RI: The dictionary does not contain duplicates and is in sorted in
      ascending order with repsect to a defined comparison function*)

    let rec extract_key_lst d =
      List.map
        (fun x ->
          match x with
          | k, _ -> k)
        d

    let rep_ok (d : t) =
      (*Nested Helper for removing duplicates for list length comparison*)
      let remove_dup ass_lst =
        List.sort_uniq
          (fun x y ->
            match
              Key.compare
                (match x with
                | kex, _ -> kex)
                (match y with
                | key, _ -> key)
            with
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
            match Key.compare h h2 with
            | LT -> is_sorted (h2 :: t)
            | GT -> false
            | EQ -> false)
      in

      let u = extract_key_lst d in
      if
        is_sorted u = true (*checks if list is sorted*)
        && List.length (extract_key_lst d) = List.length (remove_dup d)
        (*checks if list has duplicates*)
      then d
      else failwith "RI"
      [@@coverage off]

    let empty = []
    (* TODO: replace [()] with a value of your rep type [t]. Do not raise an
       exception. *)

    let is_empty (d : t) = if List.length d = 0 then true else false
    let size (d : t) = List.length d

    let rec insertion k (v : value) lst =
      match lst with
      | [] -> [ (k, v) ]
      | (ke, valu) :: t -> (
          match Key.compare k ke with
          | LT -> (k, v) :: (ke, valu) :: t
          | GT -> (ke, valu) :: insertion k v t
          | EQ -> (ke, valu) :: insertion k v t)

    let rec key_exists k lst =
      match lst with
      | [] -> false
      | h :: t -> (
          match Key.compare k h with
          | EQ -> true
          | LT -> key_exists k t
          | GT -> key_exists k t)

    let insert k v d =
      if key_exists k (extract_key_lst d) = false then insertion k v d
      else
        insertion k v
          (List.filter
             (fun x ->
               match x with
               | key, _ -> (
                   match K.compare k key with
                   | LT -> true
                   | GT -> true
                   | EQ -> false))
             d)

    let remove k d =
      if key_exists k (extract_key_lst d) = false then d
      else
        List.filter
          (fun x ->
            match x with
            | key, _ -> (
                match K.compare k key with
                | LT -> true
                | GT -> true
                | EQ -> false))
          d

    let rec search k ass_lst =
      match ass_lst with
      | [] -> None
      | h :: t -> (
          match h with
          | key, valu -> (
              match K.compare k key with
              | LT -> None
              | GT -> search k t
              | EQ -> Some valu))

    let find k d =
      if key_exists k (extract_key_lst d) = false then None else search k d

    let member k d =
      if key_exists k (extract_key_lst d) = true then true else false

    let to_list d = d

    let rec fold_helper f init d_lst =
      match d_lst with
      | [] -> init
      | (ke, valu) :: t -> fold_helper f (f ke valu init) t

    let fold f init d = fold_helper f init d

    let to_string d =
      (* Hint: use a [Util] helper function. *)
      Util.string_of_bindings Key.to_string Value.to_string (to_list d)
  end
