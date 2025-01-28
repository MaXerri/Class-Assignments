open Dictionary

module Make =
functor
  (K : KeySig)
  (V : ValueSig)
  ->
  struct
    module Key = K
    module Value = V

    type key = K.t
    type value = V.t

    (* AF: The tree Node (c,(k_i,v_i),Node (_,_,_,_),Node (_,_,_,_)) represents
       the dictionary {k1: v1, ..., (k_i,v_i), ... ,kn: vn} which maps keys to
       values. The empty tree: Leaf, represents the empty dictionary *)
    (* RI: The Red-Black tree follows the local and global invariants which are
       described in detail in the Cornell CS3110 online textbook. Also the BST
       invariant holds which can be found in the textbook as well *)

    type color =
      | Red
      | Black

    type t =
      | Leaf
      | Node of color * (key * value) * t * t

    let rep_ok d =
      (*detects if the local invariant holds*)
      let rec local_inv d =
        match d with
        | Leaf -> true
        | Node (Red, _, Node (Red, _, _, _), _) -> false
        | Node (Red, _, _, Node (Red, _, _, _)) -> false
        | Node (_, _, l, r) -> local_inv l && local_inv r
      in

      (*get the Black Height (BH) of the left-most branch to compare to other
        root leaf paths*)
      let rec bh_count_l acc tr =
        match tr with
        | Leaf -> acc
        | Node (c, _, l, r) ->
            bh_count_l
              (match c with
              | Black -> acc + 1
              | Red -> acc)
              l
      in

      (*aggregates the black count of every root leaf path*)
      let rec bh_helper acc bh tr =
        match tr with
        | Leaf -> acc = bh
        | Node (c, _, l, r) ->
            let agregate =
              match c with
              | Black -> acc + 1
              | Red -> acc
            in
            bh_helper agregate bh l && bh_helper agregate bh r
      in

      (*Checks that root to *)
      let global_inv d =
        match d with
        | Leaf -> true
        | Node (_, _, l, r) ->
            let bh_left = bh_count_l 0 l in
            bh_helper 0 bh_left l && bh_helper 0 bh_left r
      in

      if global_inv d && local_inv d then d else failwith "RI"
      [@@coverage off]

    let empty = Leaf

    let is_empty d =
      match d with
      | Leaf -> true
      | Node _ -> false

    let rec size d =
      match d with
      | Leaf -> 0
      | Node (_, _, l, r) ->
          1 + size l + size r (*citing textbook section 3.11.1*)

    let balance = function
      | Black, z, Node (Red, y, Node (Red, x, a, b), c), d
      | Black, z, Node (Red, x, a, Node (Red, y, b, c)), d
      | Black, x, a, Node (Red, z, Node (Red, y, b, c), d)
      | Black, x, a, Node (Red, y, b, Node (Red, z, c, d)) ->
          Node (Red, y, Node (Black, x, a, b), Node (Black, z, c, d))
      | a, b, c, d -> Node (a, b, c, d)

    let rec insert_aux x = function
      | Leaf -> Node (Red, x, Leaf, Leaf)
      | Node (c, v, l, r) -> (
          match
            Key.compare
              (match x with
              | ke, _ -> ke)
              (match v with
              | k, _ -> k)
          with
          | LT -> balance (c, v, insert_aux x l, r)
          | GT -> balance (c, v, l, insert_aux x r)
          | EQ ->
              Node
                ( c,
                  (match x with
                  | key, value -> (key, value)),
                  l,
                  r ))

    let insert k v d =
      match insert_aux (k, v) d with
      | Leaf -> failwith "impossible"
      | Node (_, v, l, r) -> Node (Black, v, l, r)

    let remove k d = raise (Failure "Unimplemented: TreeDictionary.Make.remove")

    let find (k : key) (d : t) =
      let rec finder k d =
        match d with
        | Leaf -> None
        | Node (_, (ke, valu), l, r) -> (
            match Key.compare k ke with
            | LT -> finder k l
            | GT -> finder k r
            | EQ -> Some valu)
      in
      finder k d

    let member k d =
      let rec mem k d =
        match d with
        | Leaf -> false
        | Node (_, (ke, valu), l, r) -> (
            match Key.compare k ke with
            | LT -> mem k l
            | GT -> mem k r
            | EQ -> true)
      in
      mem k d

    (*In order traversal function*)
    let rec in_order_trav d acc =
      match d with
      | Leaf -> acc
      | Node (color, (ke, valu), l, r) ->
          let rhs = in_order_trav r acc in
          let acc_and_val = (ke, valu) :: rhs in
          in_order_trav l acc_and_val

    let to_list d = in_order_trav d []

    let rec fold_helper f (init : 'acc) d_lst =
      match d_lst with
      | [] -> init
      | (ke, valu) :: t -> fold_helper f (f ke valu init) t

    let fold (f : key -> value -> 'acc -> 'acc) (acc : 'acc) (d : t) =
      fold_helper f acc (to_list d)

    let to_string d =
      Util.string_of_bindings Key.to_string Value.to_string (to_list d)
  end
