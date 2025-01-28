(** [StringKey] provides the necessary definitions to use strings as keys in
    dictionaries. *)

open Dictionary

module String : KeySig with type t = string = struct
  type t = string

  let compare s1 s2 =
    match Stdlib.compare s1 s2 with
    | x when x < 0 -> LT
    | x when x > 0 -> GT
    | _ -> EQ

  let to_string s = Printf.sprintf "%S" s
end

module CaselessString : KeySig with type t = string = struct
  type t = string

  let compare s1 s2 =
    match
      Stdlib.compare
        (Stdlib.String.lowercase_ascii s1)
        (Stdlib.String.lowercase_ascii s2)
    with
    | x when x < 0 -> LT
    | x when x > 0 -> GT
    | _ -> EQ

  let to_string s = Printf.sprintf "%S" (Stdlib.String.lowercase_ascii s)
end
