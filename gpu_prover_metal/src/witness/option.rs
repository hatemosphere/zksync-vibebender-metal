#[repr(C, u32)]
#[derive(Copy, Clone, Default, Debug)]
pub enum Option<T> {
    #[default]
    None,
    Some(T),
}

impl<T, U> From<core::option::Option<T>> for Option<U>
where
    T: Into<U>,
{
    fn from(option: core::option::Option<T>) -> Self {
        match option {
            Some(value) => Option::Some(value.into()),
            None => Option::None,
        }
    }
}
