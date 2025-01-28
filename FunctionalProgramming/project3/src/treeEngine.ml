(* Note that there is nothing you need to complete in this file. *)

open Dictionary

module S = DictionarySet.Make (StringKey.String) (TreeDictionary.Make)
(** [S] is a dictionary set implemented with a [TreeDictionary] whose keys are
    strings. *)

module D = TreeDictionary.Make (StringKey.CaselessString) (S)
(** [D] is a [TreeDictionary] whose keys are strings. *)

module E = Engine.Make (S) (D)
