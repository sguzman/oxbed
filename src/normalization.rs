use unicode_normalization::UnicodeNormalization;

pub fn normalize(
  input: &str
) -> String {
  let mut normalized =
    String::with_capacity(input.len());
  let mut last_was_space = false;
  for ch in input.nfkc() {
    if ch == '\r' {
      continue;
    }
    if ch.is_whitespace() {
      if !last_was_space {
        normalized.push(' ');
        last_was_space = true;
      }
      continue;
    }
    normalized.push(ch);
    last_was_space = false;
  }
  normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
  use super::normalize;

  #[test]
  fn normalize_collapses_whitespace_and_nfkc()
   {
    let raw = "\u{FB01}   bar baz";
    let normalized = normalize(raw);
    assert_eq!(
      normalized,
      "fi bar baz"
    );
  }
}
