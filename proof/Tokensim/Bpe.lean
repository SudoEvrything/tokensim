abbrev Tok := List Nat

def Ranks := Tok → Tok → Option Nat

def SuffixOf (u a : Tok) : Prop := ∃ t, a = t ++ u

def PrefixOf (v b : Tok) : Prop := ∃ t, b = v ++ t

def optMin : Option Nat → Option Nat → Option Nat
  | none, y => y
  | x, none => x
  | some x, some y => some (min x y)

theorem optMin_eq_none {x y : Option Nat} (h : optMin x y = none) :
    x = none ∧ y = none := by
  cases x with
  | none => cases y with
    | none => exact ⟨rfl, rfl⟩
    | some t => simp [optMin] at h
  | some s => cases y with
    | none => simp [optMin] at h
    | some t => simp [optMin] at h

theorem optMin_some_of_ne {x y : Option Nat} {r : Nat}
    (h : optMin x y = some r) (hx : x ≠ some r) : y = some r := by
  cases x with
  | none => exact h
  | some s =>
    cases y with
    | none =>
      simp only [optMin, Option.some.injEq] at h
      exact False.elim (hx (show some s = some r by rw [h]))
    | some t =>
      simp only [optMin, Option.some.injEq] at h
      by_cases hsle : s ≤ t
      · rw [Nat.min_eq_left hsle] at h
        exact False.elim (hx (show some s = some r by rw [h]))
      · have hts : t ≤ s := by omega
        rw [Nat.min_eq_right hts] at h
        simp [h]

theorem optMin_assoc (x y z : Option Nat) :
    optMin (optMin x y) z = optMin x (optMin y z) := by
  cases x with
  | none => rfl
  | some s =>
    cases y with
    | none => simp [optMin]
    | some t =>
      cases z with
      | none => simp [optMin]
      | some u => simp [optMin, Nat.min_assoc]

theorem optMin_some_val {y : Option Nat} {s : Nat} :
    ∃ t, optMin (some s) y = some t := by
  cases y with
  | none => exact ⟨s, rfl⟩
  | some u => exact ⟨min s u, rfl⟩

theorem optMin_some_le {y : Option Nat} {s t : Nat}
    (h : optMin (some s) y = some t) : t ≤ s := by
  cases y with
  | none => simp only [optMin, Option.some.injEq] at h; omega
  | some u => simp only [optMin, Option.some.injEq] at h; omega

def minRank (R : Ranks) : List Tok → Option Nat
  | [] => none
  | [_] => none
  | a :: b :: rest => optMin (R a b) (minRank R (b :: rest))

def mergeFirst (R : Ranks) (r : Nat) : List Tok → List Tok
  | [] => []
  | [t] => [t]
  | a :: b :: rest =>
    if R a b = some r then (a ++ b) :: rest
    else a :: mergeFirst R r (b :: rest)

def decode (l : List Tok) : Tok := l.flatMap id

theorem decode_cons (t : Tok) (l : List Tok) : decode (t :: l) = t ++ decode l := rfl

def lastTok : List Tok → Tok
  | [] => []
  | [t] => t
  | _ :: t :: ts => lastTok (t :: ts)

def headTok : List Tok → Tok
  | [] => []
  | t :: _ => t

theorem mergeFirst_decode (R : Ranks) (r : Nat) (l : List Tok) :
    decode (mergeFirst R r l) = decode l := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    cases l with
    | nil => rfl
    | cons b Bs =>
      by_cases hab : R a b = some r
      · simp only [mergeFirst, if_pos hab, decode_cons, List.append_assoc]
      · simp only [mergeFirst, if_neg hab, decode_cons, ih]

theorem mergeFirst_length_eq (R : Ranks) (r : Nat) (l : List Tok)
    (h : minRank R l = some r) : (mergeFirst R r l).length + 1 = l.length := by
  induction l with
  | nil => simp [minRank] at h
  | cons a l ih =>
    cases l with
    | nil => simp [minRank] at h
    | cons b Bs =>
      by_cases hab : R a b = some r
      · simp only [mergeFirst, if_pos hab, List.length_cons]
      · have h2 : minRank R (a :: b :: Bs) = optMin (R a b) (minRank R (b :: Bs)) := rfl
        have htail : minRank R (b :: Bs) = some r :=
          optMin_some_of_ne (by rw [← h2]; exact h) hab
        simp only [mergeFirst, if_neg hab, List.length_cons]
        rw [ih htail, List.length_cons]

theorem mergeFirst_length_lt (R : Ranks) (r : Nat) (l : List Tok)
    (h : minRank R l = some r) : (mergeFirst R r l).length < l.length := by
  have h1 := mergeFirst_length_eq R r l h
  omega

theorem minRank_append (R : Ranks) (A B : List Tok)
    (hs : R (lastTok A) (headTok B) = none) :
    minRank R (A ++ B) = optMin (minRank R A) (minRank R B) := by
  induction A with
  | nil => simp [minRank, optMin]
  | cons a A ih =>
    cases A with
    | nil =>
      cases B with
      | nil => simp [minRank, optMin]
      | cons y ys =>
        have hxy : R a y = none := hs
        simp only [List.cons_append, List.nil_append, minRank, headTok, hxy, optMin]
    | cons b Bs =>
      have ihm : minRank R (b :: (Bs ++ B)) =
          optMin (minRank R (b :: Bs)) (minRank R B) := ih hs
      simp only [List.cons_append, minRank, headTok, ihm, optMin_assoc]

theorem mergeFirst_append_left (R : Ranks) (r : Nat) (A B : List Tok)
    (h : minRank R A = some r) :
    mergeFirst R r (A ++ B) = mergeFirst R r A ++ B := by
  induction A with
  | nil => simp [minRank] at h
  | cons a A ih =>
    cases A with
    | nil => simp [minRank] at h
    | cons b Bs =>
      by_cases hab : R a b = some r
      · simp only [List.cons_append, mergeFirst, if_pos hab]
      · have h2 : minRank R (a :: b :: Bs) = optMin (R a b) (minRank R (b :: Bs)) := rfl
        have htail : minRank R (b :: Bs) = some r :=
          optMin_some_of_ne (by rw [← h2]; exact h) hab
        simp only [List.cons_append, mergeFirst, if_neg hab]
        rw [show mergeFirst R r (b :: (Bs ++ B)) =
              mergeFirst R r ((b :: Bs) ++ B) from rfl, ih htail]

theorem mergeFirst_append_right (R : Ranks) (r : Nat) (A B : List Tok)
    (hs : R (lastTok A) (headTok B) = none)
    (hA : minRank R A ≠ some r)
    (hmin : minRank R (A ++ B) = some r) :
    mergeFirst R r (A ++ B) = A ++ mergeFirst R r B := by
  induction A with
  | nil => simp only [List.nil_append]
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases B with
      | nil => simp [minRank] at hmin
      | cons y ys =>
        have hxy : R x y = none := hs
        have h3 : optMin (R x y) (minRank R (y :: ys)) = some r := by
          have h3a : minRank R (x :: y :: ys) =
              optMin (R x y) (minRank R (y :: ys)) := rfl
          have h3b : minRank R ((x :: []) ++ (y :: ys)) = some r := hmin
          rw [show minRank R ((x :: []) ++ (y :: ys)) =
                minRank R (x :: y :: ys) from rfl, h3a] at h3b
          exact h3b
        rw [hxy] at h3
        have hmb : minRank R (y :: ys) = some r := h3
        have hny : ¬(R x y = some r) := by simp [hxy]
        simp only [List.cons_append, List.nil_append, mergeFirst, if_neg hny]
    | cons b Bs =>
      have hxb : R x b ≠ some r := by
        intro hc
        have h2 : minRank R (x :: b :: Bs) = optMin (R x b) (minRank R (b :: Bs)) := rfl
        rw [hc] at h2
        have ⟨m, hmv⟩ := optMin_some_val (y := minRank R (b :: Bs)) (s := r)
        have hle2 : m ≤ r := optMin_some_le hmv
        have h4 : minRank R ((x :: b :: Bs) ++ B) = some r := hmin
        rw [minRank_append R (x :: b :: Bs) B hs, h2, hmv] at h4
        have hle1 : r ≤ m := optMin_some_le h4
        have hmr : m = r := by omega
        rw [hmv, hmr] at h2
        exact hA h2
      have hmin' : minRank R ((b :: Bs) ++ B) = some r := by
        have e1 : minRank R ((x :: b :: Bs) ++ B) =
            optMin (minRank R (x :: b :: Bs)) (minRank R B) := minRank_append _ _ _ hs
        have e2 : minRank R (x :: b :: Bs) = optMin (R x b) (minRank R (b :: Bs)) := rfl
        have e3 : minRank R ((b :: Bs) ++ B) =
            optMin (minRank R (b :: Bs)) (minRank R B) := minRank_append _ _ _ hs
        rw [e2, optMin_assoc, ← e3, hmin] at e1
        exact optMin_some_of_ne e1.symm hxb
      have hA' : minRank R (b :: Bs) ≠ some r := by
        intro hc
        have e2 : minRank R (x :: b :: Bs) = optMin (R x b) (minRank R (b :: Bs)) := rfl
        rw [e2, hc] at hA
        cases hxb' : R x b with
        | none =>
          rw [hxb'] at hA
          simp only [optMin] at hA
          exact hA rfl
        | some s =>
          rw [hxb'] at hA
          simp only [optMin] at hA
          have hminAB2 : minRank R ((x :: b :: Bs) ++ B) =
              optMin (minRank R (x :: b :: Bs)) (minRank R B) := minRank_append _ _ _ hs
          rw [e2, hxb', hc, hmin] at hminAB2
          rw [show optMin (some s) (some r) = some (min s r) from rfl] at hminAB2
          have hle : r ≤ min s r := optMin_some_le hminAB2.symm
          have hne : min s r ≠ r := by simpa using hA
          omega
      show mergeFirst R r (x :: b :: (Bs ++ B)) = (x :: b :: Bs) ++ mergeFirst R r B
      rw [mergeFirst, if_neg hxb]
      exact congrArg (fun l => x :: l) (ih hs hA' hmin')

def bpeFuel (R : Ranks) : Nat → List Tok → List Tok
  | 0, l => l
  | n + 1, l =>
    match minRank R l with
    | none => l
    | some r => bpeFuel R n (mergeFirst R r l)

theorem bpeFuel_fuel_independent (R : Ranks) :
    ∀ n l, l.length ≤ n → ∀ m, n ≤ m → bpeFuel R n l = bpeFuel R m l := by
  intro n
  induction n with
  | zero =>
    intro l hl m hm
    cases l with
    | nil =>
      cases m with
      | zero => rfl
      | succ k => simp [bpeFuel, minRank]
    | cons x xs =>
      simp only [List.length_cons] at hl
      exact absurd hl (by omega)
  | succ n ih =>
    intro l hl m hm
    cases m with
    | zero => exact absurd hm (by omega)
    | succ k =>
      cases hmin : minRank R l with
      | none => simp [bpeFuel, hmin]
      | some r =>
        have hlt : (mergeFirst R r l).length < l.length := mergeFirst_length_lt _ _ _ hmin
        have hbound : (mergeFirst R r l).length ≤ n := by omega
        simp only [bpeFuel, hmin]
        exact ih _ hbound k (by omega)

theorem decode_bpeFuel (R : Ranks) :
    ∀ n l, l.length ≤ n → decode (bpeFuel R n l) = decode l := by
  intro n
  induction n with
  | zero =>
    intro l hl
    cases l with
    | nil => rfl
    | cons x xs =>
      simp only [List.length_cons] at hl
      exact absurd hl (by omega)
  | succ n ih =>
    intro l hl
    cases hmin : minRank R l with
    | none => simp [bpeFuel, hmin]
    | some r =>
      have hlt : (mergeFirst R r l).length < l.length := mergeFirst_length_lt _ _ _ hmin
      have hbound : (mergeFirst R r l).length ≤ n := by omega
      simp only [bpeFuel, hmin]
      rw [ih _ hbound, mergeFirst_decode]

theorem bpeFuel_normal_form (R : Ranks) :
    ∀ n l, l.length ≤ n → minRank R (bpeFuel R n l) = none := by
  intro n
  induction n with
  | zero =>
    intro l hl
    cases l with
    | nil => rfl
    | cons x xs =>
      simp only [List.length_cons] at hl
      exact absurd hl (by omega)
  | succ n ih =>
    intro l hl
    cases hmin : minRank R l with
    | none => simp [bpeFuel, hmin]
    | some r =>
      have hlt : (mergeFirst R r l).length < l.length := mergeFirst_length_lt _ _ _ hmin
      have hbound : (mergeFirst R r l).length ≤ n := by omega
      simp only [bpeFuel, hmin]
      exact ih _ hbound

theorem bpeFuel_stable (R : Ranks) (n : Nat) (l : List Tok) (hl : l.length ≤ n)
    (m : Nat) : bpeFuel R m (bpeFuel R n l) = bpeFuel R n l := by
  have hnf : minRank R (bpeFuel R n l) = none := bpeFuel_normal_form R n l hl
  cases m with
  | zero => rfl
  | succ k => simp [bpeFuel, hnf]

theorem decode_last_suffix (A : List Tok) : SuffixOf (lastTok A) (decode A) := by
  induction A with
  | nil => exact ⟨[], rfl⟩
  | cons t ts ih =>
    cases ts with
    | nil => exact ⟨[], by simp [decode, lastTok]⟩
    | cons b Bs =>
      have ⟨s, hs2⟩ := ih
      refine ⟨t ++ s, ?_⟩
      rw [decode_cons, hs2, ← List.append_assoc, lastTok]

theorem decode_head_prefix (B : List Tok) : PrefixOf (headTok B) (decode B) := by
  cases B with
  | nil =>
    show ∃ t, ([] : Tok) = [] ++ t
    exact ⟨[], rfl⟩
  | cons t ts =>
    show ∃ s, (t :: ts).flatMap id = t ++ s
    exact ⟨decode ts, rfl⟩

theorem bpeFuel_append (R : Ranks) (a b : Tok)
    (hseam : ∀ u v, SuffixOf u a → PrefixOf v b → R u v = none) :
    ∀ L A B, (A ++ B).length ≤ L → decode A = a → decode B = b →
      bpeFuel R L (A ++ B) = bpeFuel R L A ++ bpeFuel R L B := by
  intro L
  induction L with
  | zero =>
    intro A B hlen hA hB
    cases A with
    | nil =>
      cases B with
      | nil => rfl
      | cons y ys =>
        simp only [List.cons_append, List.nil_append, List.length_cons] at hlen
        exact absurd hlen (by omega)
    | cons x xs =>
      simp only [List.cons_append, List.length_cons, List.length_append] at hlen
      exact absurd hlen (by omega)
  | succ L ih =>
    intro A B hlen hA hB
    cases A with
    | nil =>
      simp only [List.nil_append]
      simp [bpeFuel, minRank]
    | cons x xs =>
      cases B with
      | nil =>
        simp only [List.append_nil]
        simp [bpeFuel, minRank]
      | cons y ys =>
        have hsA : SuffixOf (lastTok (x :: xs)) a := by
          rw [← hA]; exact decode_last_suffix _
        have hsB : PrefixOf (headTok (y :: ys)) b := by
          rw [← hB]; exact decode_head_prefix _
        have hs : R (lastTok (x :: xs)) (headTok (y :: ys)) = none :=
          hseam _ _ hsA hsB
        have hminAB : minRank R ((x :: xs) ++ (y :: ys)) =
            optMin (minRank R (x :: xs)) (minRank R (y :: ys)) :=
          minRank_append _ _ _ hs
        rw [bpeFuel]
        cases hstep : minRank R ((x :: xs) ++ (y :: ys)) with
        | none =>
          have hparts : minRank R (x :: xs) = none ∧ minRank R (y :: ys) = none := by
            apply optMin_eq_none
            rw [← hminAB]
            exact hstep
          have hAx := hparts.1
          have hBy := hparts.2
          simp only [bpeFuel, hstep, hAx, hBy]
        | some r =>
          by_cases hmx : minRank R (x :: xs) = some r
          · have hstep2 : mergeFirst R r ((x :: xs) ++ (y :: ys)) =
                mergeFirst R r (x :: xs) ++ (y :: ys) :=
              mergeFirst_append_left _ _ _ _ hmx
            have hAs : (mergeFirst R r (x :: xs)).length + 1 = (x :: xs).length :=
              mergeFirst_length_eq _ _ _ hmx
            have hdec : decode (mergeFirst R r (x :: xs)) = a := by
              rw [mergeFirst_decode]; exact hA
            have hlen' : ((mergeFirst R r (x :: xs)) ++ (y :: ys)).length ≤ L := by
              have e1 : ((x :: xs) ++ (y :: ys)).length =
                  (x :: xs).length + (y :: ys).length := List.length_append
              have e2 : ((mergeFirst R r (x :: xs)) ++ (y :: ys)).length =
                  (mergeFirst R r (x :: xs)).length + (y :: ys).length :=
                List.length_append
              omega
            have hBL : (y :: ys).length ≤ L := by
              have e1 : ((x :: xs) ++ (y :: ys)).length =
                  (x :: xs).length + (y :: ys).length := List.length_append
              have e2 : ((mergeFirst R r (x :: xs)) ++ (y :: ys)).length =
                  (mergeFirst R r (x :: xs)).length + (y :: ys).length :=
                List.length_append
              omega
            have hBfuel : bpeFuel R (L + 1) (y :: ys) = bpeFuel R L (y :: ys) :=
              (bpeFuel_fuel_independent R L (y :: ys) hBL (L + 1) (by omega)).symm
            rw [hBfuel]
            simp only [bpeFuel, hstep, hmx]
            rw [hstep2]
            rw [ih (mergeFirst R r (x :: xs)) (y :: ys) hlen' hdec hB]
          · have hop : optMin (minRank R (x :: xs)) (minRank R (y :: ys)) = some r := by
              rw [← hminAB]; exact hstep
            have hmy : minRank R (y :: ys) = some r := optMin_some_of_ne hop hmx
            have hstep2 : mergeFirst R r ((x :: xs) ++ (y :: ys)) =
                (x :: xs) ++ mergeFirst R r (y :: ys) :=
              mergeFirst_append_right _ _ _ _ hs hmx hstep
            have hBs : (mergeFirst R r (y :: ys)).length + 1 = (y :: ys).length :=
              mergeFirst_length_eq _ _ _ hmy
            have hdec : decode (mergeFirst R r (y :: ys)) = b := by
              rw [mergeFirst_decode]; exact hB
            have hlen' : ((x :: xs) ++ (mergeFirst R r (y :: ys))).length ≤ L := by
              have e1 : ((x :: xs) ++ (y :: ys)).length =
                  (x :: xs).length + (y :: ys).length := List.length_append
              have e2 : ((x :: xs) ++ (mergeFirst R r (y :: ys))).length =
                  (x :: xs).length + (mergeFirst R r (y :: ys)).length :=
                List.length_append
              omega
            have hAL : (x :: xs).length ≤ L := by
              have e1 : ((x :: xs) ++ (y :: ys)).length =
                  (x :: xs).length + (y :: ys).length := List.length_append
              have e2 : ((x :: xs) ++ (mergeFirst R r (y :: ys))).length =
                  (x :: xs).length + (mergeFirst R r (y :: ys)).length :=
                List.length_append
              omega
            have hAfuel : bpeFuel R (L + 1) (x :: xs) = bpeFuel R L (x :: xs) :=
              (bpeFuel_fuel_independent R L (x :: xs) hAL (L + 1) (by omega)).symm
            rw [hAfuel]
            simp only [bpeFuel, hstep, hmy]
            rw [hstep2]
            rw [ih (x :: xs) (mergeFirst R r (y :: ys)) hlen' hA hdec]

def byteTokens (s : Tok) : List Tok := s.map (fun x => [x])

theorem byteTokens_length (s : Tok) : (byteTokens s).length = s.length := by
  simp [byteTokens]

theorem byteTokens_decode (s : Tok) : decode (byteTokens s) = s := by
  induction s with
  | nil => rfl
  | cons x xs ih =>
    have h1 : byteTokens (x :: xs) = [x] :: byteTokens xs := rfl
    rw [h1, decode_cons, ih]
    rfl

def bpeEncode (R : Ranks) (s : Tok) : List Tok := bpeFuel R s.length (byteTokens s)

theorem decode_bpeEncode (R : Ranks) (s : Tok) : decode (bpeEncode R s) = s := by
  unfold bpeEncode
  rw [decode_bpeFuel R s.length (byteTokens s) (by simp [byteTokens_length])]
  exact byteTokens_decode s

theorem bpeEncode_append (R : Ranks) (a b : Tok)
    (hseam : ∀ u v, SuffixOf u a → PrefixOf v b → R u v = none) :
    bpeEncode R (a ++ b) = bpeEncode R a ++ bpeEncode R b := by
  have hmap : byteTokens (a ++ b) = byteTokens a ++ byteTokens b := List.map_append
  unfold bpeEncode
  rw [hmap, List.length_append]
  rw [bpeFuel_append R a b hseam (a.length + b.length) (byteTokens a) (byteTokens b)
    (by simp [byteTokens, List.length_append])
    (byteTokens_decode a) (byteTokens_decode b)]
  rw [bpeFuel_fuel_independent R a.length (byteTokens a) (by simp [byteTokens_length])
    (a.length + b.length) (by omega)]
  rw [bpeFuel_fuel_independent R b.length (byteTokens b) (by simp [byteTokens_length])
    (a.length + b.length) (by omega)]

theorem decode_append (A B : List Tok) : decode (A ++ B) = decode A ++ decode B := by
  simp [decode, List.flatMap_append]

def encodeIds (R : Ranks) (idOf : Tok → Nat) (s : Tok) : List Nat :=
  (bpeEncode R s).map idOf

def decodeIds (tokBytes : Nat → Tok) (ids : List Nat) : Tok :=
  (ids.map tokBytes).flatMap id

theorem map_map_inv (tokBytes : Nat → Tok) (idOf : Tok → Nat)
    (hinv : ∀ t, tokBytes (idOf t) = t) (X : List Tok) :
    (X.map idOf).map tokBytes = X := by
  induction X with
  | nil => rfl
  | cons x xs ih => simp [List.map_cons, hinv, ih]

theorem decodeIds_encodeIds (R : Ranks) (idOf : Tok → Nat) (tokBytes : Nat → Tok)
    (hinv : ∀ t, tokBytes (idOf t) = t) (s : Tok) :
    decodeIds tokBytes (encodeIds R idOf s) = s := by
  unfold decodeIds encodeIds
  rw [map_map_inv tokBytes idOf hinv]
  exact decode_bpeEncode R s

def ranksAB : Ranks
  | [97], [98] => some 0
  | [98], [99] => some 1
  | _, _ => none

example : bpeEncode ranksAB [97, 98, 99] = [[97, 98], [99]] := by decide

example : decode (bpeEncode ranksAB [97, 98, 99]) = [97, 98, 99] := by decide

def ranksAB2 : Ranks
  | [97], [98] => some 0
  | _, _ => none

theorem seamAB2 (u v : Tok) (hu : SuffixOf u [97, 98]) (hv : PrefixOf v [99, 100]) :
    ranksAB2 u v = none := by
  have hsuf : u = [] ∨ u = [98] ∨ u = [97, 98] := by
    have ⟨t, ht⟩ := hu
    cases t with
    | nil =>
      simp only [List.nil_append] at ht
      right
      right
      exact ht.symm
    | cons x xs =>
      cases xs with
      | nil =>
        simp only [List.cons_append, List.nil_append] at ht
        injection ht with h1 h2
        subst h1
        right
        left
        exact h2.symm
      | cons y ys =>
        simp only [List.cons_append] at ht
        injection ht with h1 h2
        injection h2 with h3 h4
        subst h1
        subst h3
        cases ys with
        | nil => left; exact h4.symm
        | cons y' ys' => simp at h4
  have hpref : v = [] ∨ v = [99] ∨ v = [99, 100] := by
    have ⟨s, hs2⟩ := hv
    cases v with
    | nil => left; rfl
    | cons w ws =>
      cases ws with
      | nil =>
        simp only [List.cons_append] at hs2
        injection hs2 with h1 h2
        subst h1
        right
        left
        rfl
      | cons w2 ws2 =>
        simp only [List.cons_append] at hs2
        injection hs2 with h1 h2
        injection h2 with h3 h4
        subst h1
        subst h3
        cases ws2 with
        | nil => right; right; rfl
        | cons y' ys' => simp at h4
  cases hsuf with
  | inl h =>
    subst h
    cases hpref with
    | inl h2 => subst h2; simp [ranksAB2]
    | inr h2 => cases h2 with
      | inl h3 => subst h3; simp [ranksAB2]
      | inr h3 => subst h3; simp [ranksAB2]
  | inr h =>
    cases h with
    | inl h1 =>
      subst h1
      cases hpref with
      | inl h2 => subst h2; simp [ranksAB2]
      | inr h2 => cases h2 with
        | inl h3 => subst h3; simp [ranksAB2]
        | inr h3 => subst h3; simp [ranksAB2]
    | inr h1 =>
      subst h1
      cases hpref with
      | inl h2 => subst h2; simp [ranksAB2]
      | inr h2 => cases h2 with
        | inl h3 => subst h3; simp [ranksAB2]
        | inr h3 => subst h3; simp [ranksAB2]

example : bpeEncode ranksAB2 ([97, 98] ++ [99, 100]) =
    bpeEncode ranksAB2 [97, 98] ++ bpeEncode ranksAB2 [99, 100] :=
  bpeEncode_append ranksAB2 [97, 98] [99, 100] seamAB2
